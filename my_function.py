import numpy as np

def impute_holiday_data(data):
    data.drop(columns=['Holiday_type', 'Holiday_transferred'], inplace=True)
    data['Holiday_level'] = data['Holiday_level'].fillna(value='Not Holiday')
    data['Holiday_city'] = data['Holiday_city'].fillna(value='Not Holiday') 
    total_hol_levels = data['Holiday_level'].isnull().sum()
    total_hol_cities = data['Holiday_city'].isnull().sum()
#     data = data.drop(columns=['Holiday_type', 'Holiday_transferred'])
    print(f'The total number of missing values in Holiday_level column is {total_hol_levels}')
    print(f'The total number of missing values in Holiday_city column is {total_hol_cities}')




def payday(row):
    if row.DayOfMonth == 15 or row.Is_month_end == 1:
        return 1
    else:
        return 0
    



def type_of_day(row):
    if row.Holiday_type is np.NaN:
        if row.Is_weekend == 0:
            return 'Workday'
        else:
            return 'No work'
    elif row.Holiday_type == 'Transfer' or row.Holiday_transferred == 'True':
        return 'Transferred holiday'
    elif row.Holiday_type == 'Additional' or row.Holiday_type == 'Bridge':
        return 'Additional Holiday'
    elif row.Holiday_type == 'Work Day':
        return 'Workday'
    else:
        return row.Holiday_type
    


def impute_oil_missing_values(data):
    data['Oil_prices'] = data['Oil_prices'].interpolate(limit_direction ='backward').interpolate(method ='linear', limit_direction ='forward')
    total_missing_values = data['Oil_prices'].isnull().sum()
    print(total_missing_values)


def date_extracts(data):
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['DayOfMonth'] = data.index.day
    data['DaysInMonth'] = data.index.days_in_month
    data['DayOfYear'] = data.index.day_of_year
    data['DayOfWeek'] = data.index.dayofweek
    data['Week'] = data.index.weekofyear
    data['Is_weekend'] = np.where(data['DayOfWeek'] > 4, 1, 0)
    data['Is_month_start'] = data.index.is_month_start.astype(int)
    data['Is_month_end'] = data.index.is_month_end.astype(int)
    data['Quarter'] = data.index.quarter
    data['Is_quarter_start'] = data.index.is_quarter_start.astype(int)
    data['Is_quarter_end'] = data.index.is_quarter_end.astype(int)
    data['Is_year_start'] = data.index.is_year_start.astype(int)
    data['Is_year_end'] = data.index.is_year_end.astype(int)
