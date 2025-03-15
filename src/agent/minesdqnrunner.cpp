#include "minesdqnrunner.h"
#include <sstream>
#include <math.h>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <QImage>
#include <prism/qt/core/helper/qeventloopguard.h>
#include <prism/qt/core/helper/stopwatch.h>

namespace{

QDebug operator<<(QDebug debug, Eigen::MatrixXf value) {
    std::stringstream ss ;
    ss << std::endl;
    ss << value;
    debug << ss.str().c_str();
    return debug;
}

QDebug operator<<(QDebug debug, torch::Tensor value) {
    std::stringstream ss ;
    ss << std::endl;
    ss << value;
    debug << ss.str().c_str();
    return debug;
}

const cv::Size TARGET_SIZE(480, 256);

cv::Mat letterbox_image(const cv::Mat& image) {
    int width = image.cols;
    int height = image.rows;

    float scale = std::min(TARGET_SIZE.width / (float)width, TARGET_SIZE.height / (float)height);
    int new_width = static_cast<int>(width * scale);
    int new_height = static_cast<int>(height * scale);

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_width, new_height));

    cv::Mat padded_image(TARGET_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));
    resized_image.copyTo(padded_image(cv::Rect(0, 0, new_width, new_height)));

    return padded_image;
}

cv::Mat QImageToCvMat(const QImage &image) {
    switch (image.format()) {
    case QImage::Format_RGB32:
    {
        cv::Mat mat(image.height(), image.width(), CV_8UC4, const_cast<uchar*>(image.bits()), image.bytesPerLine());
        cv::cvtColor(mat,mat,cv::COLOR_BGRA2BGR);
        return mat.clone(); // 深拷贝以避免 QImage 数据被释放
    }
    case QImage::Format_Grayscale8: {
        cv::Mat mat(image.height(), image.width(), CV_8UC1, const_cast<uchar*>(image.bits()), image.bytesPerLine());
        return mat.clone();
    }
    default:
        qWarning("Unsupported QImage format");
        return cv::Mat();
    }
}


} //namespace

MinesDqnRunner::MinesDqnRunner(int rows,int cols,int num,bool& save_data_flag,QObject *parent) : QObject(parent),m_rows(rows),m_cols(cols),m_num(num),state_eigenMat(Eigen::MatrixXf(rows, cols)),save_data_flag(save_data_flag)
{
    thread_.reset(new QThread());
    moveToThread(thread_.get());
    thread_->start();
    this->init();

}

void MinesDqnRunner::init()
{
    this->staticMetaObject.invokeMethod(this,"private_init",Qt::QueuedConnection);
}

void MinesDqnRunner::setCurrentImage(QImage img)
{
    std::lock_guard<std::mutex> lk(imgMutex_);
    currentImg_ = img.copy();

}

void MinesDqnRunner::setEigenMat(const Eigen::MatrixXf &mat,const Eigen::MatrixXf &value_mat)
{
    std::unique_lock<std::mutex> lock(imgMutex_);
    this->state_eigenMat = mat;
    this->value_eigenMat = value_mat;
    cond_var_.notify_one();  // 通知消费者

}


void MinesDqnRunner::loadModels()
{
    // 1. 加载 TorchScript 模型
    //std::string model_path = "epo999_acc0.9998model.pt";
    std::string model_path = "model_scripted.pt";

    try {
        minesModel_ = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        qDebug()<< "模型加载失败！" ;
        return;
    }
    qDebug()<< "模型加载成功！" ;

    // 2. 设置设备为 MPS (Metal Performance Shaders)
    torch::Device device(torch::kMPS); // macOS MPS 设备
    minesModel_.to(device);
    minesModel_.eval();


}



void MinesDqnRunner::private_init()
{
    qDebug() << "MinesDqnRunner::private_init";

    loadModels();

    torch::Device device(torch::kMPS); // macOS MPS 设备


    std::string loadpath = "savepath"; //不带后辍名
    bool loaddata = true;


    policyDqn_.reset( new MinesDQN(m_rows,m_cols));
    targetDqn_.reset( new MinesDQN(m_rows,m_cols));
    optimizer_.reset(new torch::optim::Adam(policyDqn_->parameters(),learning_rate_));
    replayBuffer_.reset(new ReplayBuffer(buffer_size_)); //3步,0.99是折扣

    if(loaddata)
    {
        if(QFile::exists(QString::fromStdString(loadpath)+".pt"))
        {

            torch::NoGradGuard _;

            std::string file(loadpath+".pt");
            torch::serialize::InputArchive archive;
            archive.load_from(file);

            policyDqn_->load(archive);
            optimizer_->load(archive);

            //把policy dqn的权重复制到target dqn
            for (const auto& item : this->policyDqn_->named_parameters()) {
                this->targetDqn_->named_parameters()[item.key()].copy_(item.value());
            }

            //targetDqn_->load(archive);
        }

        if(QFile::exists("data"))
        replayBuffer_->load_from_file("data");
    }

    for (const auto& item : this->policyDqn_->named_parameters()) { item->to(device); }
    policyDqn_->to(device);

    for (const auto& item : this->targetDqn_->named_parameters()) { item->to(device); }
    targetDqn_->to(device);


}



void MinesDqnRunner::private_recognize(const QImage& img, torch::Tensor& result)
{

    //QImage inputImg = img.copy(rect);
    cv::Mat mat = QImageToCvMat(img);
    cv::Mat letterbox = letterbox_image(mat);


    //qDebug()<< "preprocess  time (ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - start).count();

    torch::Device device(torch::kMPS); // macOS MPS 设备
    letterbox.convertTo(letterbox, CV_32F, 1.0/255);
    torch::Tensor img_tensor = torch::from_blob(letterbox.data, {1,TARGET_SIZE.height, TARGET_SIZE.width, 3}).permute({0,3, 1, 2}).clone().to(device);

    result = (minesModel_.forward({img_tensor}).toTensor().argmax(-1) - 1).to(torch::kFloat) ; //-1 至 12


    //qDebug()<< "infer  time (ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - preprocess_end).count();

    //std::stringstream ss;
    //ss << "================================recognize" << std::endl;
    //for(int row = 0; row < m_rows; ++row)
    //{
    //    for(int col = 0; col<m_cols ; ++col)
    //    {
    //        ss << std::setw(4) << result[0][row][col].item<int>();

    //    }
    //    ss << std::endl;
    //}
    //ss << std::endl;
    //ss << "state.sizes:" << result.sizes() << std::endl;
    //qDebug()<< ss.str().c_str();

    //auto postprocess_end = std::chrono::high_resolution_clock::now();

    //qDebug()<< "post process time (ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(postprocess_end - infer_end).count();

}

void MinesDqnRunner::private_recognize_eigen(Eigen::MatrixXf &state_eigenMat, at::Tensor &result)
{
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> state_eigenMat_rowMajor = state_eigenMat;

    // 直接共享 Eigen 数据，避免拷贝
    result = torch::from_blob(state_eigenMat_rowMajor.data(), {1, m_rows, m_cols}, torch::kFloat).clone().to(torch::kMPS);

    //// 直接共享 Eigen 数据，避免拷贝
    //result = torch::from_blob(state_eigenMat.data(), {1, m_rows, m_cols}, torch::kFloat).to(torch::kMPS);

}

void MinesDqnRunner::private_executeAction(int action,int predictAction)
{

    int row = action / m_cols;
    int col = action % m_cols;

    int pred_row = predictAction / m_cols;
    int pred_col = predictAction % m_cols;

    if(action >=m_rows*m_cols) //重开动作
    {

        emit this->sendVncMouseEvent(-1,-1,pred_col,pred_row); //重开按钮
        //std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    else
    {

        //左上角36,255 ,每个格子26*26
        //emit this->sendVncMouseEvent(36 + 26-5 + col*26  ,255+ 26-5 + row*26, pred_col,pred_row);
        emit this->sendVncMouseEvent(col,row, pred_col,pred_row);
    }

}
int  MinesDqnRunner::selectAction(MinesDQN &net, torch::Tensor state,torch::Tensor mouse, double epsilon,int& predictAction ,int& predictPosBomb)
{
    net.eval();

    torch::Tensor q_values = net.forward(state.to(torch::kMPS).to(torch::kFloat).view({-1,1,m_rows,m_cols}),torch::tensor({{0,0}}).to(torch::kMPS));  //input: batch , channel,rows,cols, output:batch,actions

    torch::Tensor max_action = q_values.argmax(1);

    predictAction =  max_action.item<int>();



    bool  done;
    int totalcount = -1;
    done = torch::any((state == 9)|(state == 10)).item<bool>();// 9雷 10炸雷 11旗 ,都视为nf玩法的gameover


    if(done)
        return   m_rows*m_cols;
    else
    {

        totalcount = torch::sum(state == -1).item<int>() == m_num;
        done = totalcount == m_num;

        if(done)
            return   m_rows*m_cols;
    }


    static int previous_row=-1;
    static int previous_col=-1;

    if (((double)rand() / RAND_MAX) < epsilon)
    {
    retry:
        int r  =rand() % (m_rows*m_cols);
        int row = r/m_cols;
        int col = r%m_cols;
        if(previous_row != -1 && state[0][row][col].item<int>() != -1)
        {
            goto retry;
        }

        if(previous_col == col && previous_row == row)
        {
            goto retry;
        }

        previous_row = row;
        previous_col = col;

        return r;
    }
    else
    {
        if(predictAction == m_rows*m_cols)
            goto retry;
        int row = predictAction/m_cols;
        int col = predictAction%m_cols;

        return predictAction;

    }
}

void MinesDqnRunner::trainDQN(MinesDQN &policy_net, MinesDQN &target_net, ReplayBuffer &buffer, torch::optim::Adam &optimizer)
{
    STOPWATCHER(sw,"train batchs")
    policyDqn_->train(false);
    if (buffer.size() < batch_size_)
    {
        static int prebuffsize = -1;
        if(prebuffsize != buffer.size())
        {
            prebuffsize = buffer.size();
            //qDebug()<< "buffer.size:" << buffer.size();
        }
        return;  // 缓冲区未满，跳过训练
    }

    policy_net.train();
    //for(int i = 0; i< ((replayBuffer_->size() ==  buffer_size_)? smal_epoch_:1); ++i)
    for(int i = 0; i< 100; ++i)
    {
        // 采样 batch（包含 indices & weights）
        auto batch = buffer.sample(batch_size_);

        std::vector<torch::Tensor> states,mouse, actions, rewards, next_states,next_mouse;
        std::vector<int64_t> dones;  // 0 或 1 (bool 在 libtorch 中用 int64)

        for (auto &exp : batch) {
            states.push_back(exp.state);
            mouse.push_back(exp.mouse_pos);
            actions.push_back(exp.action);
            rewards.push_back(exp.reward);
            next_states.push_back(exp.next_state);
            next_mouse.push_back(exp.next_mouse_pos);
            dones.push_back(exp.done ? 1 : 0);
        }

        std::stringstream ss;

        // 转换为 tensor（转移到 MPS 加速）
        torch::Tensor state_batch = torch::stack(states).to(torch::kMPS);
        torch::Tensor mouse_batch = torch::stack(mouse).to(torch::kFloat).to(torch::kMPS);
        torch::Tensor action_batch = torch::stack(actions).to(torch::kLong).to(torch::kMPS);
        torch::Tensor reward_batch = torch::stack(rewards).to(torch::kMPS);
        torch::Tensor next_state_batch = torch::stack(next_states).to(torch::kMPS);
        torch::Tensor next_mouse_batch = torch::stack(next_mouse).to(torch::kFloat).to(torch::kMPS);
        torch::Tensor dones_tensor = torch::tensor(dones).to(torch::kMPS).to(torch::kFloat);


        // 计算 Q 估计值
        torch::Tensor q_values = policy_net.forward(state_batch.view({-1,1,m_rows,m_cols}),mouse_batch).gather(1, action_batch).squeeze(1);
        //ss << "q_values.sizes:" << q_values.sizes() <<std::endl;

        // Double DQN: 用 policy_net 选择动作，用 target_net 计算 Q 值
        torch::Tensor next_actions = policy_net.forward(next_state_batch.view({-1,1,m_rows,m_cols}),mouse_batch).argmax(1).unsqueeze(1);
        torch::Tensor next_q_values = target_net.forward(next_state_batch.view({-1,1,m_rows,m_cols}),mouse_batch).gather(1, next_actions).squeeze(1).detach();
        //ss << "next_q_values.sizes:" << next_q_values.sizes() <<std::endl;

        // 计算目标 Q 值
        torch::Tensor expected_q_values = reward_batch.squeeze() + (0.99 * next_q_values * (1 - dones_tensor));
        //ss << "reward_batch.sizes"<<reward_batch.sizes() << std::endl;
        //ss << "expected_q_values.size():" << expected_q_values.sizes() << std::endl;
        //ss << "q_values.size():" << q_values.sizes() << std::endl;

        auto loss = torch::mse_loss(q_values, expected_q_values.detach());
        // 记录 loss
        loss_ = loss.item<double>() ;
        ss << "episode:" << episode_  <<"  buffer size:" << this->replayBuffer_->size()<< "  epsilon:" << epsilon_ << "   loss :" << loss_ <<std::endl;
        qDebug()<< ss.str().c_str() ;

        //反向传播
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}



void MinesDqnRunner::train()
{
    this->staticMetaObject.invokeMethod(this,"private_train",Qt::QueuedConnection);
}


void MinesDqnRunner::private_train()
{


    for (episode_ = 0; episode_ < 10000000; ++episode_)
    {


        torch::Tensor state ;
        torch::Tensor state_copy ;
        {

            std::unique_lock<std::mutex> lock(imgMutex_);
            //private_recognize(this->currentImg_,state);// 获取扫雷游戏 m_rows 状态矩阵
            private_recognize_eigen(this->state_eigenMat,state);
        }
        state_copy = state.clone();

        bool done = torch::any((state == 9)|(state == 10)).item<bool>();// 9雷 10炸雷 11旗 ,都视为nf玩法的gameover
        if(!done)
        {
            done = torch::sum(state == -1).item<int>() == m_num;
        }


        int previous_col =-1;
        int previous_row =-1;
        int toclick = -1;
        int clicked = -1;

        int regen = true;
        while (true) {


            int row = 0;
            int col = 0;
            torch::Tensor t_numCount;

            int predictAction = 0;
            int predictPosBomb = 0;
            int a;

            bool hit;
            a= selectAction(*this->policyDqn_, state,torch::tensor({previous_row,previous_col}) ,1,predictAction,predictPosBomb);
            //a= selectAction(*this->policyDqn_, state,torch::tensor({previous_row,previous_col}) ,epsilon_,predictAction,predictPosBomb);

            if(a < m_rows * m_cols)
            {

                row = a / m_cols;
                col = a % m_cols;
                toclick = state.index({0,row,col}).item<int>();
                regen = false;
            }
            else
                regen = true;



            {
                 STOPWATCHER(sw,"send mouse event")

                 private_executeAction(a,predictAction); //发送鼠标点击
            }
            //std::this_thread::sleep_for(std::chrono::milliseconds(10000)); //等20ms抓下一个截图
            torch::Tensor next_state;
            {

                std::unique_lock<std::mutex> lock(imgMutex_);
                cond_var_.wait(lock);
                //std::lock_guard<std::mutex> lk(imgMutex_);
                //private_recognize(this->currentImg_,next_state);// 获取扫雷游戏 m_rows 状态矩阵
                private_recognize_eigen(this->state_eigenMat,next_state);

                //if(!regen)
                {
                    state_copy = state.clone();
                    int befor = torch::sum(state_copy == -1).item<int>();
                    int after = -1;
                    do
                    {
                        for(int row=0; row< m_rows;++row)
                        {
                            for (int col = 0; col< m_cols; ++col)
                            {
                                int top  = std::max(row -1 ,0);
                                int left = std::max(col -1, 0);
                                int right = std::min(col+1,m_cols-1);
                                int bottom = std::min(row+1,m_rows-1);
                                int count = 0;
                                int unknow = 0;
                                int number = state_copy[0][row][col].item<int>();
                                for(int i = top; i<= bottom;++i)
                                {
                                    for(int j = left; j<= right;++j)
                                    {
                                        int number_9_9 = state_copy[0][i][j].item<int>();
                                        if(number_9_9 == -1 || number_9_9 == 9 )
                                        {
                                            ++count;
                                        }
                                    }
                                }
                                unknow = count - number;

                                if(unknow == 0 && number!=0)
                                {
                                    for(int i = top; i<= bottom;++i)
                                    {
                                        for(int j = left; j<= right;++j)
                                        {
                                            int number_9_9 = state_copy[0][i][j].item<int>();
                                            if(number_9_9 == -1 || number_9_9 == 9 )
                                            {
                                                state_copy[0][i][j] = 9;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        after = torch::sum(state_copy == -1).item<int>();
                        if(after == befor)
                            break;
                        befor = after;
                    }
                    while(after!=befor);
                    //qDebug()<< state_copy;
                }

            }

            clicked = next_state.index({0,row,col}).item<int>();




            //torch::Tensor reward;     //(batch,1) 标量, 立即奖励值  如果开雷 -1;  如果开数字,开格数 + ( (对角线长 - 增量path) / 对角线长)
            double reward = 0;
            //path reward
            static double reward_total = 0;

            if(previous_col >=0 && previous_row >=0)
            {


                double a = row - previous_row;
                double b = col - previous_col;
                double path = std::sqrt(a * a + b * b);
                static double path_max = std::sqrt(m_rows*m_rows + m_cols*m_cols);



                if (toclick == -1) // 发现新格子
                {
                    int top  = std::max(row -1 ,0);
                    int left = std::max(col -1, 0);
                    int right = std::min(col+1,m_cols-1);
                    int bottom = std::min(row+1,m_rows-1);
                    int total = 0;
                    int count = 0;
                    int unknow = 0;
                    int unopen = 0;

                    if (clicked == 9 ||clicked == 10) //点中雷,计算雷打开的格子数量,越少减分越低
                    {
                        reward  += -1;
                    }
                    else // 打开格子,周围未知状态的雷越少,加分越多
                    {
                        std::stringstream ss;
                        for(int i = top; i<=bottom;++i)
                        {
                            for(int j = left; j<= right;++j)
                            {
                                int number_9_9 = state_copy[0][i][j].item<int>();
                                ss << number_9_9 << "  "  ;

                                if(number_9_9 == 9)
                                {
                                    ++count;
                                }
                                if(number_9_9 == -1)
                                    ++unopen;
                                ++total;
                            }
                            ss   << std::endl;
                        }
                        //qDebug()<< ss.str().c_str();
                        unknow = clicked - count ;
                        //reward  += (unopen + count) == total ?-0.3: 0.3;
                        reward  += unopen == total ?-0.1: !unknow && clicked  ?1 : -(unknow*1.0/unopen ) ;
                        //qDebug()<< "unknow:" << unknow << "  unopen:" << unopen << " reward" << reward;
                        //std::this_thread::sleep_for(std::chrono::milliseconds(3000));
                    }

                }
                else // 点中的不是未打开的格子
                {
                    //reward  += 0.3;
                    continue;
                }
                reward_total += reward;


                if( previous_col == col && previous_row == row)
                    continue;

                qDebug()<< "episode:" << episode_ << "  pos:" << row << "," << col << " [" << toclick <<"->" << clicked  << "] pred_pos:"
                         << predictAction/m_cols << "," << predictAction%m_cols  <<  "reward:" << reward << "/" << reward_total;

            }


            this->replayBuffer_->push(state,
                                      torch::tensor({previous_row,previous_col}),
                                      torch::tensor({a}),
                                      torch::tensor(reward),
                                      next_state,
                                      torch::tensor({row,col}),
                                      done );


            state = next_state;
            previous_col = col;
            previous_row = row;

            done = torch::any((state == 9)|(state == 10)).item<bool>();// 9雷 10炸雷 11旗 ,都视为nf玩法的gameover
            if(!done)
            {
                done = torch::sum(state == -1).item<int>() == m_num;
            }




            if(this->replayBuffer_->size() >= 10000)
            //if(this->replayBuffer_->size() >= buffer_size_)
            //if(this->replayBuffer_->size() >= batch_size_)
            {
                trainDQN(*this->policyDqn_, *this->targetDqn_, *this->replayBuffer_, *this->optimizer_);
            }


            if(done)
            {
                qDebug()<< "buffer size:" <<this->replayBuffer_->size();

                break;
            }

        }




        // 更新 target 网络
        if (episode_ && episode_ % 1 == 0)
        {

            torch::NoGradGuard _;


            //python中的写法 target_net.load_state_dict(policy_net.state_dict());  不适用于c++,改写为下面的for循环
            for (const auto& item : this->policyDqn_->named_parameters()) {
                this->targetDqn_->named_parameters()[item.key()].copy_(item.value());
            }


            //std::string model_path = QString("savepath").arg(episode_).arg(loss_).toStdString();
            std::string model_path = QString("savepath").toStdString();//.arg(episode_).arg(loss_).toStdString();
            //保存模型
            {
                torch::serialize::OutputArchive output_archive;
                this->policyDqn_->save(output_archive);
                this->optimizer_->save(output_archive);
                output_archive.save_to(model_path+".pt");
            }

            epsilon_ = std::max(0.1, epsilon_ * 0.9);  // 逐步减少探索概率
        }

        if(this->save_data_flag)
        {
            this->replayBuffer_->save_to_file("data");
        }


    }


}



