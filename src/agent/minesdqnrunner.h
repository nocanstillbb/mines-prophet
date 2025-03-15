#ifndef MINESDQNRUNNER_H
#define MINESDQNRUNNER_H

#include <condition_variable>
#include <QDebug>
#include <QEventLoop>
#include <QImage>
#include <QObject>
#include <QThread>
#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <vector>
#include <eigen3/Eigen/Dense>


#undef slots
#include <torch/script.h>
#include <torch/nn/functional.h>
#include <torch/optim.h>
#include <torch/optim/adam.h>
#include <torch/torch.h>
#define slots Q_SLOTS

using json = nlohmann::json;

struct MinesDQN ;
struct Experience;
class ReplayBuffer;

class MinesDqnRunner : public QObject
{
    Q_OBJECT
public:
    explicit MinesDqnRunner(int rows,int cols,int num,bool& save_data_flag,QObject *parent = nullptr);

    void init();
    void pushData(torch::Tensor state, torch::Tensor action, torch::Tensor reward, torch::Tensor next_state, bool done);

public slots:
    void setCurrentImage(QImage img);
    void setEigenMat(const Eigen::MatrixXf &mat,const Eigen::MatrixXf &value_mat);

    void train();

private slots:
    void private_init();
    void loadModels();
    void private_recognize(const QImage& img, torch::Tensor& result);
    void private_recognize_eigen(Eigen::MatrixXf& state_eigenMat, torch::Tensor& result);
    void private_executeAction(int action,int predictAction);
    void trainDQN(MinesDQN &policy_net, MinesDQN& target_net, ReplayBuffer &buffer, torch::optim::Adam &optimizer);

    void private_train();

    int selectAction(MinesDQN &net, torch::Tensor state,torch::Tensor mouse, double epsilon,int& predictAction ,int& predictPosBomb);

signals:
    void sendVncMouseEvent(int x ,int y, int pred_x,int pred_y);

private:
    int batch_size_ = 128;
    double buffer_size_ = 30000;
    double learning_rate_ = 0.001;
    int smal_epoch_ = 1000;
    double epsilon_ = 1; // 探索率 越高越探索,越低则越倾向于模型预测的结果
    int episode_ = 0;
    double loss_ = 0;

    bool& save_data_flag;

    int m_rows = 16;
    int m_cols = 30;
    int m_num = 100;
    Eigen::MatrixXf state_eigenMat;
    Eigen::MatrixXf value_eigenMat;

    torch::jit::script::Module minesModel_;
    std::unique_ptr<QThread> thread_;
    std::shared_ptr<MinesDQN> policyDqn_;
    std::shared_ptr<MinesDQN> targetDqn_;
    std::shared_ptr<torch::optim::Adam> optimizer_;
    std::shared_ptr<ReplayBuffer> replayBuffer_;

    QImage currentImg_;


    std::mutex imgMutex_;
    std::condition_variable cond_var_;


};
struct MinesDQN : torch::nn::Module {
private:
    int m_rows = 9;
    int m_cols = 9;
public:
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Conv2d conv3{nullptr};
    torch::nn::Conv2d conv4{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};

    MinesDQN(int rows, int cols) : m_rows(rows), m_cols(cols) {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 128, 3).stride(1).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)));
        conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)));

        int flattened_size = 128 * m_rows * m_cols ; //

        // 线性层
        fc1 = register_module("fc1", torch::nn::Linear(flattened_size, 512));
        fc2 = register_module("fc2", torch::nn::Linear(512, 512));
        fc3 = register_module("fc3", torch::nn::Linear(512, m_rows * m_cols));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor mouse_pos) {

        x = x/13; //-1 - 12的值 归一化

        if(this->is_training())
        {
            // 获取 batch size
            int batch_size = x.size(0);
            int height = x.size(2);
            int width = x.size(3);

            // **创建注意力 Mask**
            torch::Tensor mask = torch::zeros({batch_size, 1, height, width}, x.options());

            for (int i = 0; i < batch_size; i++) {
                int row = mouse_pos[i][0].item<int>();
                int col = mouse_pos[i][1].item<int>();

                // 确保 5×5 区域在边界内
                int top = std::max(0, row - 2);
                int bottom = std::min(height, row + 3);
                int left = std::max(0, col - 2);
                int right = std::min(width, col + 3);

                // 设置 Mask 为 1
                mask[i].slice(1, top, bottom).slice(2, left, right) = 1;
            }

            // **应用 Mask**
            x = x * mask + x.detach() * (1 - mask);  // Mask 之外区域 detach
            //x = x * mask ;  // Mask 之外区域 detach
        }



        // **卷积层**
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = torch::relu(conv3->forward(x));
        x = torch::relu(conv4->forward(x));


        // **展开后通过全连接层**
        x = x.view({x.size(0), -1});
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);

        return x;
    }


};





struct Experience {
    torch::Tensor state;      //(batch,m_rows,m_cols)
    torch::Tensor mouse_pos;    //(batch,2) 当前鼠标位置
    torch::Tensor action;     //(batch,1)   范围0 ~ m_rows*m_cols  最大值表示点击某个格子或点重开按钮
    torch::Tensor reward;     //(batch,1) 标量, 立即奖励值  如果开雷 -1;  如果开数字,开格数 * ( (对角线长 - 增量path) / 对角线长)
    torch::Tensor next_state; //(batch,m_rows,m_cols)
    torch::Tensor next_mouse_pos; // (batch,2)下一个鼠标位置
    bool done;                // 游戏是否结束
};


// 经验回放缓冲区
struct ReplayBuffer {
    std::random_device rd;
    std::mt19937 gen;
    std::deque<Experience> buffer;
    size_t capacity;
    ReplayBuffer(int cap) :capacity(cap), gen(rd()) {}
    void push(torch::Tensor state, torch::Tensor mouse, torch::Tensor action, torch::Tensor reward,
              torch::Tensor next_state, torch::Tensor next_mouse, bool done) {

            Experience exp = {state,
                              mouse,
                              action,
                              reward,
                              next_state,
                              next_mouse,
                              done};

            if (buffer.size() >= capacity) {
                buffer.pop_front();
            }
            buffer.push_back(exp);
    }

    std::vector<Experience> sample(int batch_size) {
        std::vector<Experience> result ;
        result.reserve(batch_size);
        std::sample(buffer.begin(), buffer.end(), std::back_inserter(result), batch_size, gen);
        return result;
    }
    int size()
    {
        return buffer.size();
    }

    void save_to_file(const std::string &filename) {
        torch::serialize::OutputArchive archive;
        size_t buffer_size = buffer.size();

        archive.write("buffer_size", torch::tensor(static_cast<int64_t>(buffer_size)));

        for (size_t i = 0; i < buffer_size; ++i) {
            archive.write("state_" + std::to_string(i), buffer[i].state);
            archive.write("mouse_pos_" + std::to_string(i), buffer[i].mouse_pos);
            archive.write("action_" + std::to_string(i), buffer[i].action);
            archive.write("reward_" + std::to_string(i), buffer[i].reward);
            archive.write("next_state_" + std::to_string(i), buffer[i].next_state);
            archive.write("next_mouse_pos_" + std::to_string(i), buffer[i].next_mouse_pos);
            archive.write("done_" + std::to_string(i), torch::tensor(static_cast<int64_t>(buffer[i].done)));
        }

        archive.save_to(filename);
    }
    void load_from_file(const std::string &filename) {
        torch::serialize::InputArchive archive;
        try {
            archive.load_from(filename);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading replay buffer: " << e.what() << std::endl;
            return;
        }

        torch::Tensor buffer_size_tensor;
        archive.read("buffer_size", buffer_size_tensor);
        size_t buffer_size = buffer_size_tensor.item<int64_t>();

        buffer.clear();
        for (size_t i = 0; i < buffer_size; ++i) {
            Experience exp;
            archive.read("state_" + std::to_string(i), exp.state);
            archive.read("mouse_pos_" + std::to_string(i), exp.mouse_pos);
            archive.read("action_" + std::to_string(i), exp.action);
            archive.read("reward_" + std::to_string(i), exp.reward);
            archive.read("next_state_" + std::to_string(i), exp.next_state);
            archive.read("next_mouse_pos_" + std::to_string(i), exp.next_mouse_pos);

            torch::Tensor done_tensor;
            archive.read("done_" + std::to_string(i), done_tensor);
            exp.done = static_cast<bool>(done_tensor.item<int64_t>());

            buffer.push_back(exp);
        }
    }

};


#endif // MINESDQNRUNNER_H
