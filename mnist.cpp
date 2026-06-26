#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>

struct NeuralNet : torch::nn::Module {
    torch::nn::Linear layer1{ nullptr }, layer2{ nullptr };

    NeuralNet(int64_t in, int64_t hid, int64_t out) {
        layer1 = register_module("layer1", torch::nn::Linear(in, hid));
        layer2 = register_module("layer2", torch::nn::Linear(hid, out));

        torch::nn::init::kaiming_uniform_(layer1->weight, std::sqrt(5.0));
        torch::nn::init::kaiming_uniform_(layer2->weight, std::sqrt(5.0));
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

cv::Mat tensor_to_mat(const torch::Tensor& img_tensor) {
    auto img = img_tensor.clone().view({ 28, 28 });
    img = img * 0.3081 + 0.1307;
    img = torch::clamp(img, 0.0, 1.0);
    img = (img * 255.0).to(torch::kUInt8);
    cv::Mat mat(28, 28, CV_8UC1);
    std::memcpy(mat.data, img.data_ptr(), 28 * 28);
    cv::Mat resized;
    cv::resize(mat, resized, cv::Size(280, 280), 0, 0, cv::INTER_NEAREST);
    cv::Mat color;
    cv::cvtColor(resized, color, cv::COLOR_GRAY2BGR);
    return color;
}

int main() {
    try {
        const int64_t input_size = 784, hidden_size = 500, num_classes = 10;
        const int64_t batch_size = 100;
        const double lr = 0.001;
        const int num_epochs = 10;

        // 使用相对路径，前提是 exe 在项目根目录（D:\codeAI\DNN_v2_0\DNN）运行
        std::string data_root = "./data";

        auto train_ds = torch::data::datasets::MNIST(data_root, torch::data::datasets::MNIST::Mode::kTrain);
        auto test_ds = torch::data::datasets::MNIST(data_root, torch::data::datasets::MNIST::Mode::kTest);
        int64_t n_train = train_ds.size().value(), n_test = test_ds.size().value();
        std::cout << "Training samples: " << n_train << "\nTest samples: " << n_test << std::endl;

        // 保存前5个训练样本为 BMP
        std::cout << "Saving first 5 training samples...\n";
        for (int i = 0; i < 5; ++i) {
            auto ex = train_ds.get(i);
            auto img = ex.data;
            int64_t lbl = ex.target.item<int64_t>();
            cv::Mat mat = tensor_to_mat(img);
            cv::putText(mat, "Label: " + std::to_string(lbl), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            std::string filename = "train_sample_" + std::to_string(i) + ".bmp";
            cv::imwrite(filename, mat);
            system(("start " + filename).c_str());
        }

        NeuralNet net(input_size, hidden_size, num_classes);
        auto criterion = torch::nn::CrossEntropyLoss();
        torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(lr));

        std::random_device rd;
        std::mt19937 rng(rd());

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
                    auto img = ex.data;
                    auto lbl = ex.target.unsqueeze(0);
                    img = (img - 0.1307) / 0.3081;
                    img = img.view({ 1, -1 });
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

        net.eval();
        torch::NoGradGuard no_grad;
        int64_t correct = 0, total = 0;
        for (int64_t s = 0; s < n_test; s += batch_size) {
            int64_t e = std::min(s + batch_size, n_test);
            std::vector<torch::Tensor> imgs, lbls;
            for (int64_t i = s; i < e; ++i) {
                auto ex = test_ds.get(i);
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
        std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) << acc << "%\n";

        std::cout << "Saving test samples with predictions...\n";
        for (int i = 0; i < 20; ++i) {
            auto ex = test_ds.get(i * 500);
            auto img_raw = ex.data;
            int64_t true_lbl = ex.target.item<int64_t>();
            auto img_norm = (img_raw - 0.1307) / 0.3081;
            auto pred = net.forward(img_norm.view({ 1, -1 })).argmax(1).item<int64_t>();

            cv::Mat mat = tensor_to_mat(img_raw);
            cv::putText(mat, "True: " + std::to_string(true_lbl), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::putText(mat, "Pred: " + std::to_string(pred), cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            std::string filename = "test_sample_" + std::to_string(i) + ".bmp";
            cv::imwrite(filename, mat);
            system(("start " + filename).c_str());
        }
        std::cout << "Done. Check the generated BMP files." << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "LibTorch error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}