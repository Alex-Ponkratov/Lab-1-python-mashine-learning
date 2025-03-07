import pandas as pd
from sklearn.preprocessing import MinMaxScaler
df_train = pd.read_csv("train.csv")

print("Первые строки датасета:")
print(df_train)

# 2. Проверка пропущенных значений
print("\nКоличество пропущенных значений в каждом столбце:")
print(df_train.isnull().sum())

# 3. Заполнение пропущенных значений
df_train.fillna({
    'Age': df_train['Age'].median(),
    'Cabin': df_train['Cabin'].mode()[0],
    'HomePlanet': df_train['HomePlanet'].mode()[0]
}, inplace=True)

# 4. Нормализация числовых данных
scaler = MinMaxScaler()
numerical_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])

# 5. Преобразование категориальных данных
df_train = pd.get_dummies(df_train, columns=['HomePlanet'], drop_first=True)

# 6. Сохранение обработанных данных
df_train.to_csv("processed_titanic_new.csv", index=False)
print("\nОбработанный датасет сохранен в 'processed_titanic_new.csv'")