import numpy as np
from sklearn.linear_model import LinearRegression
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156]
def data_form (inputs, param1, param2):
    output=np.array(inputs)
    output=np.reshape(output,(param1,param2))
    return output

def regression(x, y, xtest, ytest):
    ln = LinearRegression()
    ln.fit(x, y) #fit() - обучение модели линейной регрессии
    print('Значение коэффициента при х =', ln.coef_)
    print('Значения отклонения =', ln.intercept_) #выводим значения отклонения
    ypred = ln.predict(xtest) #прогноз на 12 месяц
    mape = (abs(ypred - ytest)) / ytest  # Рассчитываем среднюю относительную ошибку МАРЕ
    print('Коэффициент детерминации =', ln.score(x, y))  # Выводим коэффициент детерминации - оценку качества линейной модели
    print('MAPE =',mape)
    return ypred

regression(data_form(months [0:9], -1, 1), data_form(revenue [0:9], -1, 1), data_form(months[10], -1, 1), revenue [10])
