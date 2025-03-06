import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Загрузка данных
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_sample_submission = pd.read_csv("sample_submission.csv")


print("Первые строки датасета:")
print(df_train.head())

# Заполнение пропущенных значений
df_train.fillna({
    'Age': df_train['Age'].median(),
    'Cabin': df_train['Cabin'].mode()[0],
    'HomePlanet': df_train['HomePlanet'].mode()[0]
}, inplace=True)

# Нормализация числовых данных
scaler = MinMaxScaler()
df_train['Age'] = scaler.fit_transform(df_train[['Age']])

#Преобразование категориальных данных (one-hot encoding)
df_train = pd.get_dummies(df_train, columns=['HomePlanet'], drop_first=True)

# Сохранение обработанных данных
df_train.to_csv("processed_titanic.csv", index=False)
# Пример объединения датафреймов
df_combined = pd.concat([df_train, df_test, df_sample_submission], ignore_index=True)
print("Обработанный датасет сохранен в 'processed_titanic.csv'")