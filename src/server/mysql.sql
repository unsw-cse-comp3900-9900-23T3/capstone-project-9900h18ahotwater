CREATE DATABASE Comp9900;
USE Comp9900;
CREATE TABLE IF NOT EXISTS `user`(
    `user_id` INT UNSIGNED AUTO_INCREMENT,
    `nickname` VARCHAR(50) NOT NULL,
    `username` VARCHAR(40) NOT NULL,
    `password` VARCHAR(40) NOT NULL,
    `email` VARCHAR(40) NOT NULL,
    `phone` VARCHAR(40) NOT NULL,
   PRIMARY KEY ( `user_id` )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `history`(
    `history_id` INT UNSIGNED AUTO_INCREMENT,
    `user_id` INT UNSIGNED NOT NULL,
    `data_id` INT UNSIGNED NOT NULL,
    `history_time` DATETIME NOT NULL,
    `history_model` VARCHAR(40) NOT NULL,
    `classes` VARCHAR(256) NOT NULL,
    `probability` VARCHAR(256) NOT NULL,
    PRIMARY KEY ( `history_id` )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE IF NOT EXISTS `data`(
    `data_id` INT UNSIGNED AUTO_INCREMENT,
    `num_img` INT UNSIGNED NOT NULL,
    `img1` VARCHAR(100) NOT NULL,
    `img2` VARCHAR(100) NOT NULL,
    `text` VARCHAR(300) NOT NULL,
    PRIMARY KEY ( `data_id` )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `user`( `nickname`, `username`, `password`, `email`, `phone`) VALUES ('admin','admin','123456','admin@unsw.edu.au','0432123456');
INSERT INTO `user`( `nickname`, `username`, `password`, `email`, `phone`) VALUES ('user','user','000000','user@qq.com','1234556');

INSERT INTO `data`( `num_img`, `img1`, `img2`, `text`) VALUES (2,'data/1.jpg','data/2.jpg','They are people');
INSERT INTO `data`( `num_img`, `img1`, `img2`, `text`) VALUES (1,'data/3.jpg','data/3.jpg','there is two cars in the picture. One is a car and the other is a truck. the car is a taxi.');

Insert into `history`(`user_id`, `data_id`, `history_time`, `history_model`, `classes`, `probability`) VALUES (1,2,'2023-10-10 10:10:10','SFSC','car,truck','0.8448,0.7819');
Insert into `history`(`user_id`, `data_id`, `history_time`, `history_model`, `classes`, `probability`) VALUES (2,1,'2023-10-10 10:10:10','SFSC','person','0.8448');