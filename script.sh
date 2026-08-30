#!/bin/bash

sudo dnf install -y python3.12 python3.12-pip python3.12-devel git nginx
cd /home/ec2-user/
git clone https://github.com/Siddheswara/mlops.git
cd mlops

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "dvc[s3]"
dvc pull
rm -f artifacts/return_model.joblib
pip install "mlflow>=2.14,<3"
python -m returns_app.train --data returns_dataset.csv --output artifacts/return_model.joblib --no-mlflow
sudo tee /etc/systemd/system/returns_app.service > /dev/null <<'EOF'
[Unit]
Description=Returns Prediction FastAPI (ASGI/Uvicorn)
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/mlops
Environment=MODEL_PATH=/home/ec2-user/mlops/artifacts/return_model.joblib
Environment=PATH=/home/ec2-user/mlops/.venv/bin:/usr/bin
ExecStart=/home/ec2-user/mlops/.venv/bin/uvicorn returns_app.api:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/nginx/conf.d/returns_api.conf > /dev/null <<'EOF'
upstream returns_uvicorn {
    server 127.0.0.1:8000;
    keepalive 32;
}
limit_req_zone $binary_remote_addr zone=predict_limit:10m rate=10r/s;
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;
    client_max_body_size 2m;
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;
    gzip_min_length 256;
    location / {
        proxy_pass http://returns_uvicorn;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        proxy_read_timeout 60s;
    }
    location /predict {
        limit_req zone=predict_limit burst=20 nodelay;
        proxy_pass http://returns_uvicorn;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        proxy_read_timeout 60s;
    }
}
EOF

sudo systemctl daemon-reload
sudo systemctl enable returns_app.service
sudo systemctl start returns_app.service
sudo systemctl enable nginx
sudo systemctl start nginx

