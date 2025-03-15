#include "gameveiwmodel.h"
#include <prism/qt/core/helper/stopwatch.h>
#include <iostream>
#include <sstream>
#include <random>
#include <set>
#include <QtConcurrent/QtConcurrent>
#include <QDebug>

namespace  {

QDebug operator<<(QDebug debug, Eigen::MatrixXf value) {
    std::stringstream ss ;
    ss << std::endl;
    ss << value;
    debug << ss.str().c_str();
    return debug;
}


}

// 3x3 卷积函数
Eigen::MatrixXf GameVeiwmodel::conv3x3(const Eigen::MatrixXf& input, const Eigen::MatrixXf& kernel) {
    int kSize = 3;  // 卷积核大小
    int pad = 1;    // 填充大小

    // **创建填充后的矩阵** (Zero Padding)
    Eigen::MatrixXf padded = Eigen::MatrixXf::Zero(rows() + 2 * pad, cols() + 2 * pad);
    padded.block(pad, pad, rows(), cols()) = input;

    // 输出矩阵的大小 (Valid 卷积)
    Eigen::MatrixXf output(rows() , cols() );

    // 遍历输入矩阵
    for (int i = 0; i < rows(); ++i) {
        for (int j = 0; j < cols(); ++j) {
            // 取出 3×3 区域
            Eigen::MatrixXf patch = padded.block(i, j, kSize, kSize);

            // 计算卷积（逐元素相乘并求和）
            output(i, j) = (patch.array() * kernel.array()).sum();
        }
    }
    return output;
}

GameVeiwmodel::GameVeiwmodel(QObject *parent) : QObject(parent)
{
    initCells(); //初始化
    regen(); //生成雷


    QObject::connect(this->minesRunner_.get(), &MinesDqnRunner::sendVncMouseEvent,this,[this](int x ,int y,[[maybe_unused]] int pred_x,[[maybe_unused]]int pred_y){

        STOPWATCHER(sw,"execute mouse event")
        if(x == -1 && y == -1)
        {
            this->regen();
            qDebug()<<  "regend";
        }
        else
        {
            this->open(y*m_cols + x);
            //qDebug()<<  "opened row:" << y << "  cols:" << x;
        }

        setClickedIndex(y* m_cols + x);
        setPredictIndex(pred_y* m_cols + pred_x);

        this->minesRunner_->setEigenMat(m_visualMat,m_minesMat);
    });

}


void GameVeiwmodel::initCells()
{

    m_kernel = Eigen::MatrixXf(3, 3);
    m_kernel << 1, 1, 1, 1, 0, 1, 1, 1, 1;

    m_minesMat = Eigen::MatrixXf(this->rows(), this->cols());
    m_visualMat = Eigen::MatrixXf(this->rows(), this->cols());

    prism::qt::core::prismModelListProxy<CellInfo>* cells = new prism::qt::core::prismModelListProxy<CellInfo>(this);
    cells->pub_beginResetModel();
    for(int i =0; i< rows() * cols(); ++i)
    {
        std::shared_ptr<CellInfo> row = std::make_shared<CellInfo>();
        cells->appendItemNotNotify(row);
    }
    cells->pub_endResetModel();
    setCells(cells);
}

void GameVeiwmodel::open(int index)
{

    int row = index/cols();
    int col = index%cols();

    int v = m_minesMat(row,col);


    if(v == 9)//开到雷,把所有格子打开,把当前格子设置为10(红色的雷)
    {

        if(m_isFirst)
        {

            regen();

            open(index);

            return;
        }
        m_visualMat = m_minesMat;
        m_visualMat(row,col) = 10;

    }
    else if(v == 0)
    {

        recurse_open(index);
    }
    else
    {

        if(m_isFirst)
        {

            regen();

            open(index);

            return;
        }
        m_visualMat(row,col) = m_minesMat(row,col);


    }

    bool isFinished = false;

    int unopened = ((m_visualMat.array().cast<int>() == -1).cast<int>()).sum();

    int mines = ((m_minesMat.array().cast<int>() == 9).cast<int>()).sum();


    if(unopened ==  mines)
    {

        isFinished = true;
    }


    for(int row =0; row<rows();++row)
    {
        for(int col= 0; col<cols(); ++col)
        {

            std::shared_ptr<prism::qt::core::prismModelProxy<CellInfo>> cell = cells()->list()->at(row*cols() + col);
            cell->instance()->visual_value = m_visualMat(row,col);
            if(isFinished && cell->instance()->value == 9)
                cell->instance()->visual_value = 11;
            cell->update();
        }
    }



    //if(m_isFirst)
    //{
    //    std::stringstream ss;
    //    for(int i=0; i< rows(); ++i)
    //    {
    //        for(int j =0; j< cols(); ++j)
    //        {
    //            ss << std::setw(4) << cells()->list()->at(i*cols() + j)->instance()->value;
    //        }
    //        ss << std::endl;
    //    }
    //    qDebug()<< ss.str().c_str();
    //}

    m_isFirst = false;

}


void GameVeiwmodel::regen()
{
    m_isFirst = true;

    std::random_device rd;  // 用于获取随机种子
    std::mt19937 gen(rd()); // Mersenne Twister 生成器
    std::uniform_int_distribution<int> dist(0, rows()*cols() -1); // 生成 [0,m_rows * m_cols] 之间的整数

    m_minesMat.setConstant(-1);
    m_visualMat.setConstant(-1);


    for(std::shared_ptr<prism::qt::core::prismModelProxy<CellInfo>> cell : *this->cells()->list())
    {
        cell->instance()->value = -1;
        cell->instance()->visual_value = -1;
    }

    std::set<int> values;
    for(int i = 0; i<num(); ++i)
    {
        retry:
        int index = dist(gen); //0-m_rows * m_cols
        if(values.find(index) == values.end())
        {
            values.insert(index);
            this->m_cells->list()->at(index)->instance()->value = 9;
            m_minesMat(index/cols(),index%cols()) = 9;
        }
        else
            goto retry;
    }

    //qDebug()<< "result  before :" << minesMat;

    Eigen::MatrixXf result = conv3x3((m_minesMat.array() ==9).cast<float>()+
                                     (m_minesMat.array() ==10).cast<float>(),
                                     m_kernel);
    //qDebug()<< "result  after :" << result;

    for(int i =0; i< rows() * cols(); ++i)
    {
        std::shared_ptr<prism::qt::core::prismModelProxy<CellInfo>> cell = cells()->list()->at(i);
        int row = i/cols();
        int col = i%cols();
        int v = result(row,col);
        if(cell->instance()->value == -1)
        {
            cell->instance()->value = v;
            cell->instance()->visual_value = -1;
            m_minesMat(row,col) = v;
        }
        cell->update();
    }

}


void GameVeiwmodel::recurse_open(int i)
{
    int row = i/cols();
    int col = i%cols();

    m_visualMat(row,col) = 0;

    Eigen::MatrixXf mask = Eigen::MatrixXf(this->rows(), this->cols());
    mask.setConstant(0);
    mask(row,col) = 1;
    mask = conv3x3(mask, m_kernel);
    //mask = mask.array() * (minesMat.array() ==0).cast<float>();
    m_visualMat =  mask.select(m_minesMat,m_visualMat);


    std::set<int> diff;
    //通知ui
    for(int j =0; j< rows() * cols(); ++j)
    {
        std::shared_ptr<prism::qt::core::prismModelProxy<CellInfo>> cell = cells()->list()->at(j);
        int v = m_visualMat(j/cols(),j%cols());
        if(cell->instance()->visual_value != v)
        {
            cell->instance()->visual_value = v;
            if(i != j && v ==0)
                diff.insert(j);
        }
    }
    for(int item : diff)
    {
        recurse_open(item);
    }


}


prism::qt::core::prismModelListProxy<CellInfo> *GameVeiwmodel::cells() const
{
    return m_cells;
}

void GameVeiwmodel::setCells(prism::qt::core::prismModelListProxy<CellInfo> *newCells)
{
    if (m_cells == newCells)
        return;
    m_cells = newCells;
    emit cellsChanged();
}

int GameVeiwmodel::rows() const
{
    return m_rows;
}

void GameVeiwmodel::setRows(int newRows)
{
    if (m_rows == newRows)
        return;
    m_rows = newRows;
    emit rowsChanged();
}

int GameVeiwmodel::cols() const
{
    return m_cols;
}

void GameVeiwmodel::setCols(int newCols)
{
    if (m_cols == newCols)
        return;
    m_cols = newCols;
    emit colsChanged();
}

int GameVeiwmodel::num() const
{
    return m_num;
}

void GameVeiwmodel::setNum(int newNum)
{
    if (m_num == newNum)
        return;
    m_num = newNum;
    emit numChanged();
}

void GameVeiwmodel::trainDnq()
{
    //QtConcurrent::run([this](){
    //    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    //    this->minesRunner_->setEigenMat(m_visualMat);
    //});
    this->minesRunner_->train();

}

int GameVeiwmodel::predictIndex() const
{
    return m_predictIndex;
}

void GameVeiwmodel::setPredictIndex(int newPredictIndex)
{
    if (m_predictIndex == newPredictIndex)
        return;
    m_predictIndex = newPredictIndex;
    emit predictIndexChanged();
}

int GameVeiwmodel::clickedIndex() const
{
    return m_clickedIndex;
}

void GameVeiwmodel::setClickedIndex(int newClickedIndex)
{
    if (m_clickedIndex == newClickedIndex)
        return;
    m_clickedIndex = newClickedIndex;
    emit clickedIndexChanged();
}

bool GameVeiwmodel::save_data_flag() const
{
    return m_save_data_flag;
}

void GameVeiwmodel::setSave_data_flag(bool newSave_data_flag)
{
    if (m_save_data_flag == newSave_data_flag)
        return;
    m_save_data_flag = newSave_data_flag;
    emit save_data_flagChanged();
}
