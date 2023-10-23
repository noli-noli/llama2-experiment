FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

#docker-composeから引数を受け取る
ARG http_tmp
ARG https_tmp

#環境変数にプロキシの設定を追加
ENV http_proxy=$http_tmp
ENV https_proxy=$https_tmp

#タイムゾーンを東京に設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#apt関連の設定
RUN apt update -y && apt upgrade -y
RUN apt install -y nano screen tmux systemd init nginx python3 python3-pip sed

#nginxの設定
RUN sed -i "s/listen 80 default_server/listen 8080 default_server/g" /etc/nginx/sites-enabled/default 
RUN sed -i "s/listen \[\:\:\]\:80 default_server/listen \[\:\:\]\:8080 default_server/g" /etc/nginx/sites-enabled/default 
RUN sed -i "s/root \/var\/www\/html/root \/workspace\/front/g" /etc/nginx/sites-enabled/default