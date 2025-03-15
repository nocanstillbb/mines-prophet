#ifndef GAMEVEIWMODEL_H
#define GAMEVEIWMODEL_H

#include <QVariantMap>
#include <QObject>
#include <prism/qt/core/hpp/prismQt.hpp>
#include <prism/qt/core/hpp/prismModelListProxy.hpp>
#include <eigen3/Eigen/Dense>

struct CellInfo{
    /**
     * @brief value 真实状态
     * -1   : 未打开
     * 0    : 空
     * 1-8  : 数字
     * 9-10 : 雷
     * 11   : 旗
     * 12   : 问号
     */
    int value = -1;

    /**
     * @brief visual_value 可视状态
     * -1   : 未打开
     * 0    : 空
     * 1-8  : 数字
     * 9-10 : 雷
     * 11   : 旗
     * 12   : 问号
     */
    int visual_value = -1;

    /**
     * @brief isPressed
     * 双击旗时,未打开的格子凹下
     */
    bool isPressed = false;

    /**
     * @brief isLastPressed
     * 点击到雷
     */
    bool isLastPressed = false;
};
PRISMQT_CLASS(CellInfo)
PRISM_FIELDS(CellInfo,value,visual_value,isPressed,isLastPressed)


class GameVeiwmodel : public QObject
{
    Q_OBJECT

    Q_PROPERTY(prism::qt::core::prismModelListProxy<CellInfo>* cells READ cells WRITE setCells NOTIFY cellsChanged)
    Q_PROPERTY(int rows READ rows WRITE setRows NOTIFY rowsChanged)
    Q_PROPERTY(int cols READ cols WRITE setCols NOTIFY colsChanged)
    Q_PROPERTY(int num READ num WRITE setNum NOTIFY numChanged)
    Q_PROPERTY(int predictIndex READ predictIndex WRITE setPredictIndex NOTIFY predictIndexChanged)

    Q_PROPERTY(int clickedIndex READ clickedIndex WRITE setClickedIndex NOTIFY clickedIndexChanged)

    Q_PROPERTY(bool save_data_flag READ save_data_flag WRITE setSave_data_flag NOTIFY save_data_flagChanged)

public:
    explicit GameVeiwmodel(QObject *parent = nullptr);





    prism::qt::core::prismModelListProxy<CellInfo> *cells() const;
    void setCells(prism::qt::core::prismModelListProxy<CellInfo> *newCells);

    int rows() const;
    void setRows(int newRows);

    int cols() const;
    void setCols(int newCols);

    int num() const;
    void setNum(int newNum);

    int predictIndex() const;
    void setPredictIndex(int newPredictIndex);

    int clickedIndex() const;
    void setClickedIndex(int newClickedIndex);

    bool save_data_flag() const;
    void setSave_data_flag(bool newSave_data_flag);

private:
    Eigen::MatrixXf conv3x3(const Eigen::MatrixXf& input, const Eigen::MatrixXf& kernel);
    void initCells();
    void recurse_open(int i);
    prism::qt::core::prismModelListProxy<CellInfo> *m_cells = nullptr;

public slots:
    void regen();
    void open(int i);
private:

    Eigen::MatrixXf m_kernel;
    Eigen::MatrixXf m_minesMat;
    Eigen::MatrixXf m_visualMat;
    //高级
    //int m_rows = 16;
    //int m_cols = 30;
    //int m_num = 100;

    int m_rows = 9;
    int m_cols = 9;
    int m_num = 10;

    bool m_isFirst = true;



    int m_predictIndex = -1;

    int m_clickedIndex = -1;

    bool m_save_data_flag =false;

signals:

    void cellsChanged();
    void rowsChanged();
    void colsChanged();
    void numChanged();

    void clicked(int x ,int y,int p_x ,int p_y);

    void predictIndexChanged();
    void clickedIndexChanged();
    void save_data_flagChanged();
};

#endif // GAMEVEIWMODEL_H
