# 开启web服务:
```shell
python manage.py runserver 0.0.0.0:8000
```
# 访问web: 
[http://localhost:8000/app/](http://localhost:8000/app/)

# 开启rasa server
```shell
rasa run -m models --enable-api --cors "*" --debug
```

# 开启 rasa actions server
```shell
rasa run actions
```