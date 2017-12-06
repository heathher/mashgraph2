#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"

//
#include "matrix.h"
typedef Matrix<std::tuple<float, float, float>> Image;
//

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

//приведение изображение к квадратному размеру по бОльшей стороне
Image ImageToSquare(Image src_image){
    uint col = src_image.n_cols;
    uint row = src_image.n_rows;
    uint new_col = col > row ? col : row;
    Image result(new_col, new_col);
    for (uint i = 0; i < row; ++i ){
        for (uint j = 0; j < col; ++j){
            float r, g, b;
            std::tie(r, g, b) = src_image(i, j);
            result(i, j) = std::make_tuple(r, g, b);
        }
    }
    if (new_col == col){
        for (uint i = row; i < new_col; ++i){
            for (uint j = 0; j < new_col; ++j){
                result(i, j) = std::make_tuple(0, 0, 0);
            }
        }
    } else {
        for (uint i = 0; i < row; ++i){
            for (uint j = col; j < new_col; ++j){
                result(i, j) = std::make_tuple(0, 0, 0);
            }
        }
    }
    return result;
}

//BMP to Image
Image BMPtoImage(BMP * src_image){
    uint row = src_image->TellHeight();
    uint col = src_image->TellWidth();
    Image result(row, col);
    for (uint i = 0; i < row; ++i){
        for (uint j = 0; j < col; ++j){
            float r, g, b;
            r = src_image->GetPixel(j, i).Red;
            g = src_image->GetPixel(j, i).Green;
            b = src_image->GetPixel(j, i).Blue;
            result(i, j) = std::make_tuple(r, g, b);
        }
    }
    return result;
}

//изменение размера изображения в scale раз (из задания №1)
Image resize(Image src_image, double scale) {
    uint row = (src_image.n_rows - 1) * scale , col = (src_image.n_cols - 1) * scale ;
    Image res_image(row, col);
    uint r1, g1, b1, r2, g2, b2;

    for (uint i = 0; i < row; ++i){
        for (uint j = 0; j < col; ++j){
            //билинейное интерполирование
            double new_i = i / scale;
            double new_j = j / scale;
            int y = (j/scale);
            int x = (i/scale);
            double alpha = new_i - x;
            std::tie(r1, g1, b1) = src_image(x + 1, y);
            std::tie(r2, g2, b2) = src_image(x, y);
            std::tuple<double, double, double> value1 = std::make_tuple(r1 * alpha + r2 * (1 - alpha), 
                                                                        g1 * alpha + g2 * (1 - alpha), 
                                                                        b1 * alpha + b2 * (1 - alpha));
            std::tie(r1, g1, b1) = src_image(x + 1, y + 1);
            std::tie(r2, g2, b2) = src_image(x, y + 1);
            std::tuple<double, double, double> value2 = std::make_tuple(r1 * alpha + r2 * (1 - alpha), 
                                                                        g1 * alpha + g2 * (1 - alpha), 
                                                                        b1 * alpha + b2 * (1 - alpha));
            double beta = new_j - y;
            std::tie(r1, g1, b1) = value2;
            std::tie(r2, g2, b2) = value1;
            res_image(i, j) = std::make_tuple(r1 * beta + r2 * (1 - beta), 
                                                g1 * beta + g2 * (1 - beta),
                                                b1 * beta + b2 * (1 - beta)); 
        }
    }
    return res_image;
}

//получение матрицы яркости
Matrix<float> ToGrayscale (Image src_image){
    uint row = src_image.n_rows;
    uint col = src_image.n_cols;
    Matrix<float> result(row, col);
    for (size_t i = 0; i < row; ++i){
        for (size_t j = 0; j < col; ++j){
            float r, g, b;
            std::tie(r, g, b) = src_image(i, j);
            result(i, j) = (0.299*r + 0.587*g + 0.114*b ) ;        
        }
    }
    return result;
}

//применение ядра к изображению (из задания №1)
Matrix<float> Custom(Matrix<float> src_image, Matrix<float> kernel){
    //for kernel with radius = 1
    Matrix<float> res_image = src_image.deep_copy(); 
    for (uint i = 1; i < src_image.n_rows - 1; i++){
        for (uint j = 1; j < src_image.n_cols - 1; j++){
            float sum = 0;
            for (int k = -1; k < 2; k++){
                for (int l = -1; l < 2; l++){
                    sum += src_image(i + k, j + l) * kernel(k + 1, l + 1);
                }
            }
            res_image(i, j) = sum;
        } 
    }
    return res_image;
}

//получение длины вектора производной
Matrix<float> GetMod(Matrix<float> x, Matrix<float> y){
    uint row = x.n_rows;
    uint col = x.n_cols;
    Matrix<float> result(row, col);
    for (uint i = 0; i < row; ++i){
        for (uint j = 0; j < col; ++j){
            result(i, j) = pow(pow(x(i, j), 2) + pow(y(i, j), 2), 0.5);
        }
    }
    return result;
}

//получение угла направления вектора производной
Matrix<float> GetAngle(Matrix<float> x, Matrix<float> y){
    uint row = x.n_rows;
    uint col = x.n_cols;
    Matrix<float> result(row, col);
    for (uint i = 0; i < row; ++i){
        for (uint j = 0; j < col; ++j){
            //получим угол от 0 до 360
            result(i, j) = atan2(y(i, j), x(i, j) * 180.0 / 3.14159265) + 180;
        }
    }
    return result;
}

//take histogram of oriented gradients
std::vector<float> TakeHistogram(uint segsize, Matrix<float> mod, Matrix<float> angle){
    std::vector<float>result(segsize, 0);
    uint row = mod.n_rows;
    uint col = mod.n_cols;
    for (uint i = 0; i < row; ++i){
        for (uint j = 0; j < col; ++j){
            float seg_num = floor(angle(i, j) / (360/segsize));
            result[seg_num] += mod(i, j);
        }
    }
    return result;
}
//нормализуем гистограмму
std::vector<float> NormHistogram(std::vector<float> hist){
    uint size = hist.size();
    std::vector<float> result = hist;
    float norma = 0;
    for (uint i = 0; i < size; ++i){
        norma += pow(hist[i], 2); 
    }
    norma = pow(norma, 0.5);
    if (norma > 0){
        for (uint i = 0; i < size; ++i){
            if (norma > 0)
                result[i] = hist[i] / norma; 
        }
    }
    return result;
}
//доп.задание №1: сравнение пикселя по яркости с его соседями
std::vector<int> CmpNbhd(int x, int y, Matrix<float> src_image){
    std::vector<int> result(8, 0);
    //если яркость пикселя больше яркости соседа, то записываем 0, иначе - 1
    if (x >= 1 && y >= 1){
        result[0] = src_image(x-1, y-1) > src_image(x, y) ? 1 : 0;
        result[1] = src_image(x-1, y) > src_image(x, y) ? 1 : 0;
        result[2] = src_image(x-1, y+1) > src_image(x, y) ? 1 : 0;
        result[3] = src_image(x, y+1) > src_image(x, y) ? 1 : 0;
        result[4] = src_image(x+1, y+1) > src_image(x, y) ? 1 : 0;
        result[5] = src_image(x+1, y) > src_image(x, y) ? 1 : 0;
        result[6] = src_image(x+1, y-1) > src_image(x, y) ? 1 : 0;
        result[7] = src_image(x, y-1) > src_image(x, y) ? 1 : 0;
    }
    return result;
}

//доп.задание №1: приведение двоичного числа к десятиричному 
int BinaryToDec(std::vector<int> binary){
    int result = 0;
    for (uint i = 0; i < binary.size(); ++i){
        result = result << 1;
        result += binary[i];
    }
    return result;
}

//доп.задание №1: дескриптор локальных бинарных шаблонов
std::vector<float> LBPDesc(Matrix<float> src_image){
    std::vector<float> result(256, 0);
    int row = src_image.n_rows;
    int col = src_image.n_cols;
    for (int i = 1; i < row - 1; ++i){
        for (int j = 0; j < col - 1; ++j){
            //сравним пиксель по яркости с его соседями
            std::vector<int> cmpnbhd_result = CmpNbhd(i, j, src_image);
            //приведем полученный вектор из 0 и 1 к десятичному числу
            int dec_number = BinaryToDec(cmpnbhd_result);
            //прибавим единицу в соответсвующий элемент вектора
            result[dec_number] ++;
        }
    }
    return result;
}
//доп.задание №2: цветовые признаки
//возьмем среднее значение по каждому каналу и поделим на 255
std::vector<float> ColorFeatures(Image src_image, float size){
    uint row = src_image.n_rows;
    uint col = src_image.n_cols;
    std::vector<float> result(3, 0);
    for (uint i = 0; i < row; ++i){
        for (uint j = 0; j < col; ++j){
            float r, g, b;
            std::tie(r, g, b) = src_image(i, j);
            result[0] += r;
            result[1] += g;
            result[2] += b;
        }
    }
    result[0] /= (size*255);
    result[1] /= (size*255);
    result[2] /= (size*255);
    return result;
}
// Exatract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    Matrix<float> sobel_x = {{-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1}};
    Matrix<float> sobel_y = {{ 1, 2, 1},
                             { 0, 0, 0},
                             {-1,-2,-1}};
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        std::vector<float> result_hist(0, 0);
        
        // PLACE YOUR CODE HERE
        //приведем BMP к Image 
        Image src_image = BMPtoImage(data_set[image_idx].first);
        //сделаем изображение квадратным
        src_image = ImageToSquare(src_image);
        //посчитаем во сколько раз нужно изменить изображение
        //чтобы привести к размеру 64х64
        double scale =  64.0 / (src_image.n_rows - 1);
        //изменим размер изображение в scale раз
        src_image = resize(src_image, scale);
        //получим матрицу яркости нашего изображения
        Matrix<float> cur_image = ToGrayscale(src_image);
        //применим вертикального Собеля
        Matrix<float> res_sobel_x = Custom(cur_image, sobel_x);
        //применим горизонтального Собеля
        Matrix<float> res_sobel_y = Custom(cur_image, sobel_y);
        //найдем длины векторов производный
        Matrix<float> vector_mod = GetMod(res_sobel_x, res_sobel_y);
        //найдем направление векторов производных
        Matrix<float> vector_angle = GetAngle(res_sobel_x, res_sobel_y);
        //на какое количество клеточек в ряд делим изображение?
        uint cell_count = 8;
        for (uint i = 0; i < cell_count; ++i){
            for (uint j = 0; j < cell_count; ++j){
                //количество сегментов на которое мы делим 360 градусов
                uint segment_size = 6;
                //получим гистограмму по направлению и длине векторов производных
                std::vector<float> tmp_hist = TakeHistogram(segment_size, 
                        vector_mod.submatrix(8*i, 8*j, 8, 8), vector_angle.submatrix(8*i, 8*j, 8, 8));
                //доп.задание №1: получим дескриптор локальных бинарных шаблонов
                std::vector<float> desc_hist = LBPDesc(cur_image.submatrix(8*i, 8*j, 8, 8));
                //конкатенируем оба вектора 
                tmp_hist.insert(tmp_hist.end(), desc_hist.begin(), desc_hist.end());
                //нормализуем вектор
                std::vector<float> normal_hist = NormHistogram(tmp_hist);
                result_hist.insert(result_hist.end(), normal_hist.begin(), normal_hist.end());
                //доп.задание №2: получим вектор из цветовых признаков
                std::vector<float> color_features = ColorFeatures(src_image.submatrix(8*i, 8*j, 8, 8), 64.0);
                result_hist.insert(result_hist.end(), color_features.begin(), color_features.end());
            }
        }
        features->push_back(make_pair(result_hist, data_set[image_idx].second));
    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.25;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}