# 使用方法
```shell
python app.py
```

# 依赖
```shell
flask==2.0.2
itsdangerous==2.0.1
Jinja2==3.0.3
MarkupSafe==2.0.1
```

# REST

REST channel  使用POST方法访问下面的url
http://<host>:<port>/webhooks/rest/webhook,
参数格式为:
{
  "sender": "test_user",  // sender ID of the user sending the message
  "message": "Hi there!"
}

The response from Rasa Open Source will be a JSON body of bot responses,
for example:
[
  {"text": "Hey Rasa!"}, {"image": "http://example.com/image.jpg"}
]


# 访问方法
127.0.0.1:5000