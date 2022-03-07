import configparser
import os
def find_config_data():
    dir = "E:\\Works\\Project\\Students_and_Certificates_Comparison\\CharacterIdentification" + "\\config.ini"
    conf = configparser.ConfigParser()
    conf.read(dir)
    return conf
def get_dir_test_image():
    conf = find_config_data()
    test_dir = conf.get("data_dir","test_dir")
    images_name = os.listdir(test_dir)
    return test_dir, images_name
def get_dir_opencvmodel():
    conf = find_config_data()
    dir = conf.get("model_dir", "opencv_model_dir")
    model_name = os.listdir(dir)
    return dir, model_name
def get_dir_Chinese_Characters():
    '''->dir, files_under_this_dir'''
    conf = find_config_data()
    dir = conf.get("data_dir", "chinese_character")
    files = os.listdir(dir)
    return dir, files
def a():
    config = find_config_data()
    config.add_section("data")

    dir = os.getcwd() + "\\config.ini"
    with open(dir,"w+") as f:
        config.write(f)

