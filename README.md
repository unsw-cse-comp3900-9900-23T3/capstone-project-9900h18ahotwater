# COMP9900 H18A Hotwater
SRUM master:
* Zhenghao Yu (z5412301)
Group members:
* Haibo Huang (z5390552)
* Yiming Ren  (z5383693)
* Dixuan Liu  (z5232340)
* Jinzhao Li  (z5193638)

## Introduction

pass

## How to install

1. Clone the project to your local machine
2. Install python3.11.3
3. pip install -r requirements.txt
4. install mysql
5. use mysql to import the database file in the project (mysql.sql), you can use the following command:
    ```shell
    #login mysql
    mysql -u root -p 
    #import database
    source /path_to/mysql.sql
    ```
6. download weights to src/models/weights, you can find the download link in the readme.md file in the folder
7. (optional) download the dataset to src/models/dataset, you can find the download link in the readme.md file in the folder

## How to run
1. to run the server, you can use the following command:
    ```shell
    python3 app.py
    ```

2. to run the train script, you can use the following command:
    ```shell
    python3 train.py --data_path coco --epochs 100 --model_name coco1 --batch_size 16
    ```



## Branches

* main: the main branch for the project, nobody can push to this branch directly
* master: the branch for solving conflicts. Before you push your code to main branch, you should merge your code to this branch first for solving the conflicts, and then merge this branch to main branch
* diary: the branch for diary, used for weekly diary
* feature-xxx: the branch for xxx feature, used for xxx feature (for example)
* bugfix-xxx: the branch for xxx bugfix, used for xxx bugfix (for example)
* test-xxx: the branch for xxx test, used for xxx test (for example)
* doc-xxx: the branch for xxx document, used for xxx document (for example)
* to be continued...