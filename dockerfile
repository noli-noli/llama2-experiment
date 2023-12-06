#FROM nvidia/cuda:11.4.3-base-ubuntu20.04
FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

#docker-composeから環境変数の引数を受け取る
ARG http_tmp
ARG https_tmp

#環境変数を環境に設定
ENV http_proxy=$http_tmp
ENV https_proxy=$https_tmp

#llama-cppでgpuを使うための環境変数
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE=1

#タイムゾーンを東京に設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#apt関連の設定
RUN apt update -y && apt upgrade -y
RUN apt install -y nano screen tmux systemd init nginx python3 python3-pip sed

#pip関連の設定
RUN mkdir -p /tmp
COPY ./requirements.txt /tmp/requirements.txt
WORKDIR /tmp
RUN pip install -r requirements.txt

#nginxの設定
#RUN sed -i "s/listen 80 default_server/listen 8080 default_server/g" /etc/nginx/sites-enabled/default 
#RUN sed -i "s/listen \[\:\:\]\:80 default_server/listen \[\:\:\]\:8080 default_server/g" /etc/nginx/sites-enabled/default 
#RUN sed -i "s/root \/var\/www\/html/root \/workspace\/front/g" /etc/nginx/sites-enabled/default