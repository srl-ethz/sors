#ifndef PARAMS_H
#define PARAMS_H

#include "common.h"

/**
 * @class Params
 * @brief Key–value store for simulation parameters.
 *
 * The Params class wraps a string-to-MatrixXd dictionary used to pass
 * scalar and vector parameters around the codebase.
 * Internally all values are stored as Eigen::MatrixXd, but a set of
 * convenience overloads allows setting parameters from:
 * - double / int scalars
 * - std::vector<double> / std::vector<int> (treated as column vectors)
 * - MatrixXd directly
 * - CSV files containing key, value1, value2, ...
 *
 * Typical usage:
 * @code
 * Params p;
 * p.set_param("density", 1000.0);
 * p.set_param("gravity", (MatrixXd(3,1) << 0, 0, -9.81).finished());
 * MatrixXd g = p.get_value("gravity");
 * @endcode
 *
 * The class performs basic consistency checks (sizes of existing entries,
 * matching key/value array lengths) via assertions.
 */
class Params{

public:
    // Constructor from vectors of keys and MatrixXd values
    Params(std::vector<std::string> keys, std::vector<MatrixXd> values) {
        assert(keys.size() == values.size());
        auto key = keys.begin();
        auto value = values.begin();
        for(; key != keys.end(); ++key, ++value){
            set_param(*key, *value);
        }
    };

    // Constructor from vectors of keys and double values
    Params(std::vector<std::string> keys, std::vector<double> values) {
        assert(keys.size() == values.size());
        auto key = keys.begin();
        auto value = values.begin();
        for (; key != keys.end(); ++key, ++value) {
            set_param(*key, *value);
        }
    };

    Params(){};

    // Retrieve the value associated with a key
    MatrixXd get_value (std::string key) {
        if (dict_.find(key)==dict_.end()) {
            std::cerr << "Key not found in the parameter list: " << key << std::endl;
            assert(dict_.find(key)!=dict_.end());
        }
        return this->dict_[key];
    }

    // Set the MatrixXd value for a given key
    void set_param (std::string key, MatrixXd value) {
        if (dict_.find(key)!=dict_.end()) {
            assert(dict_[key].size() == value.size());
            dict_[key] = value;
        } else dict_.insert(std::make_pair(key, value));
    }

    // Set the std::vector<double> value for a given key
    void set_param (std::string key, std::vector<double> values) {
        // It must be a MatrixXd shaped (N, 1)
        if (dict_.find(key)!=dict_.end()) {
            assert(dict_[key].rows() == static_cast<Eigen::Index>(values.size()));
            for (size_t i = 0; i < values.size(); ++i) {
                dict_[key](i, 0) = values[i];
            }
        } else {
            Eigen::Map<Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrixValue(values.data(), values.size(), 1);
            dict_.insert(std::make_pair(key, matrixValue));
        }
    }

    // Set the double value for a given key
    void set_param (std::string key, double value) {
        if (dict_.find(key)!=dict_.end()) {
            assert(dict_[key].size() == 1);
            dict_[key](0, 0) = value;
        } else {
            MatrixXd matrixValue = (MatrixXd(1, 1) << value).finished();
            dict_.insert(std::make_pair(key, matrixValue));
        }
    }

    // Set the std::vector<int> value for a given key
    void set_param (std::string key, std::vector<int> values) {
        // It must be a MatrixXd shaped (N, 1)
        if (dict_.find(key)!=dict_.end()) {
            assert(dict_[key].rows() == static_cast<Eigen::Index>(values.size()));
            for (size_t i = 0; i < values.size(); ++i) {
                dict_[key](i, 0) = static_cast<double>(values[i]);
            }
        } else {
            std::vector<double> valuesD(values.begin(), values.end());
            Eigen::Map<Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrixValue(valuesD.data(), valuesD.size(), 1);
            dict_.insert(std::make_pair(key, matrixValue));
        }
    }

    // Set the int value for a given key
    void set_param (std::string key, int value) {
        if (dict_.find(key)!=dict_.end()){
            assert(dict_[key].size() == 1);
            dict_[key](0, 0) = static_cast<double>(value);
        } else {
            MatrixXd matrixValue = (MatrixXd(1, 1) << static_cast<double>(value)).finished();
            dict_.insert(std::make_pair(key, matrixValue));
        }
    }

    // Load parameters from a file
    void set_param (const std::string& filename) {
        if (!loadFile(filename)) std::cout << "Failed to load the parameter file." << std::endl;
        assert(loadFile(filename));
    }

private:
    std::map<std::string, MatrixXd> dict_; 

    // Helper method to parse a line from the CSV file
    void parseCSVLine(const std::string &line) {
        std::vector<std::string> result;
        std::stringstream sstream(line);
        std::string cell;
        while (std::getline(sstream, cell, ',')) {
            result.push_back(cell);
        }
        assert(result.size() >= 2);
        std::string key = result[0];
        if (result.size() > 2) {
            std::vector<double> value(result.size() - 1, 0);
            for (size_t i = 0; i < value.size(); ++i)
                value[i] = std::stod(result[i + 1]);
            set_param(key, value);
        }
        else {
            set_param(key, std::stod(result[1]));
        }
        return;
    }

    // Method to load parameters from a CSV file
    bool loadFile(const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Could not open the file: " << filename << std::endl;
            return false;
        }
        if (filename.substr(filename.find_last_of(".") + 1) == "csv") {
            std::string line;
            while (std::getline(file, line))
            {
                parseCSVLine(line);
            }
        }
        else {
            std::cerr << "Could not load this type of file: " << filename << std::endl;
            return false;
        }
        file.close();
        return true;
    }
};

#endif