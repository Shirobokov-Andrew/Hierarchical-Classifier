# Введение
### В данной папке находятся файлы для иерархического классификатора:
### *./amazon* - папка с датасетом
### *hierarchical_classifier.py* - исходный код иерархического классификатора.
### *classifier_fast_api.py* - FastAPI с использованием обученного иерархического классификатора
### *main.py* - программа, в которой я проводил профайлинг и небольшие тесты.
### *profile_output.txt* - результаты профайлинга с помощью line_profiler
### *hierarchical_classifier_notebook.ipynb* - Jupyter notebook, в котором содержатся различные эксперименты с иерархическим и плоским классификатором с подробным текстовым описанием.
### *Dockerfile* - докер файл для запуска FastAPI приложения
### *requirements.txt* - необходимые зависимости для сборки Docker-образа и запуска Docker-контейнера.

# Иерархический классификатор
### Я написал простой иерархический классификатор, который моделирует распределение вида: 
```
P(Cat1, Cat2, Cat3) = P(Cat3 | Cat1, Cat2)*P(Cat2 | Cat1)*P(Cat1)
```
### Более подробное описание работы данного класификатора можно найти в соответствующем файле *hierarchical_classifier.py*.
### Эксперименты с иерархическим классификатором и сравнение его с плоским классификатором можно найти в jupyter-ноутбуке *hierarchical_classifier_notebook.ipynb*.

# FastAPI
В файле *classifier_fast_api.py* содерижится небольшое FastAPI для взаимодействия с обученным иерархическим классификатором - на вход принимает post-запрос с текстом в виде JSON, а на выходе получаются категории вида:
```
"Cat1": cat_1_model_prediction,
"Cat2": cat_2_model_prediction,
"Cat3": cat_3_model_prediction
```
Для отдельной проверки я запускал его на локальном хосте с помощью:
```bash
uvicorn classifier_fast_api:app --host 0.0.0.0 --port 8000
```
А затем в десктопной версии Postman посылал запрос по адресу:
```
http://127.0.0.1:8000/predict/
```
Подавая на вход JSON вида:
```json
{
    "text": "The description and photo on this product needs to be changed to indicate this product is the BuffalOs version of this beef jerky."
}
```

# Dockerfile
### В ***Dockerfile*** содержится установка всех необходимых зависимостей, указанных в ***requirements.txt***.
### Находясь в директории со всеми необходимыми файлами, я собрал докер-образ просто с помощью:
```bash
docker build -t my-fastapi-app .
```
### А затем запустил контейнер на локальном хосте с помощью:
```bash
docker run -d -p 8000:8000 my-fastapi-app
```
### После чего запросы можно посылать точно так же с помощью Postman по схеме, указанной выше.