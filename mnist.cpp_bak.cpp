#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

struct NeuralNet : torch::nn::Module {
    torch::nn::Linear layer1{ nullptr }, layer2{ nullptr };

    NeuralNet(int64_t in, int64_t hid, int64_t out) {
        layer1 = register_module("layer1", torch::nn::Linear(in, hid));
        layer2 = register_module("layer2", torch::nn::Linear(hid, out));

        // 应用与 PyTorch 一致的 Kaiming 初始化
        torch::nn::init::kaiming_uniform_(layer1->weight, std::sqrt(5.0));
        torch::nn::init::kaiming_uniform_(layer2->weight, std::sqrt(5.0));
        // bias 初始化为 0（PyTorch 默认也是 0）
        if (layer1->bias.defined()) torch::nn::init::constant_(layer1->bias, 0.0);
        if (layer2->bias.defined()) torch::nn::init::constant_(layer2->bias, 0.0);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = layer1->forward(x);
        x = torch::relu(x);
        x = layer2->forward(x);
        return x;
    }
};

int main() {
    try {
        const int64_t input_size = 784, hidden_size = 500, num_classes = 10;
        const int64_t batch_size = 100;
        const double lr = 0.001;
        const int num_epochs = 15;      // 足够收敛到 96%+

        // 加载 MNIST（数据已归一化到 [0,1]，float）
        auto train_ds = torch::data::datasets::MNIST("./data", torch::data::datasets::MNIST::Mode::kTrain);
        auto test_ds = torch::data::datasets::MNIST("./data", torch::data::datasets::MNIST::Mode::kTest);
        int64_t n_train = train_ds.size().value(), n_test = test_ds.size().value();
        std::cout << "Training samples: " << n_train << "\nTest samples: " << n_test << std::endl;

        NeuralNet net(input_size, hidden_size, num_classes);
        auto criterion = torch::nn::CrossEntropyLoss();
        torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(lr));

        std::random_device rd;
        std::mt19937 rng(rd());

        // 训练循环
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            net.train();
            std::vector<int64_t> indices(n_train);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);

            double running_loss = 0.0;
            int64_t batches = 0;
            for (int64_t s = 0; s < n_train; s += batch_size) {
                int64_t e = std::min(s + batch_size, n_train);
                std::vector<torch::Tensor> imgs, lbls;
                for (int64_t i = s; i < e; ++i) {
                    auto ex = train_ds.get(indices[i]);
                    auto img = ex.data;                  // [1,28,28] float
                    auto lbl = ex.target.unsqueeze(0);   // 变成 1 维
                    img = (img - 0.1307) / 0.3081;       // 标准化
                    img = img.view({ 1, -1 });             // 展平
                    imgs.push_back(img);
                    lbls.push_back(lbl);
                }
                auto X = torch::cat(imgs, 0);
                auto Y = torch::cat(lbls, 0);
                optimizer.zero_grad();
                auto loss = criterion(net.forward(X), Y);
                loss.backward();
                optimizer.step();
                running_loss += loss.item<double>();
                ++batches;
            }
            std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
                << " - avg loss: " << std::fixed << std::setprecision(4)
                << (running_loss / batches) << std::endl;
        }

        // 评估函数
        auto evaluate = [&](torch::data::datasets::MNIST& ds, int64_t n, const char* name) {
            net.eval();
            torch::NoGradGuard no_grad;       // 关闭梯度计算
            int64_t correct = 0, total = 0;
            for (int64_t s = 0; s < n; s += batch_size) {
                int64_t e = std::min(s + batch_size, n);
                std::vector<torch::Tensor> imgs, lbls;
                for (int64_t i = s; i < e; ++i) {
                    auto ex = ds.get(i);
                    auto img = ex.data;
                    auto lbl = ex.target.unsqueeze(0);
                    img = (img - 0.1307) / 0.3081;
                    img = img.view({ 1, -1 });
                    imgs.push_back(img);
                    lbls.push_back(lbl);
                }
                auto X = torch::cat(imgs, 0);
                auto Y = torch::cat(lbls, 0);
                auto preds = net.forward(X).argmax(1);
                total += Y.size(0);
                correct += preds.eq(Y).sum().item<int64_t>();
            }
            double acc = 100.0 * correct / total;
            std::cout << name << " accuracy: " << std::fixed << std::setprecision(2)
                << acc << "%" << std::endl;
            return acc;
            };

        evaluate(train_ds, n_train, "Train");
        evaluate(test_ds, n_test, "Test");
    }
    catch (const c10::Error& e) {
        std::cerr << "LibTorch error: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}