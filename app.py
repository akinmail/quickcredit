import pandas as pd
from flask import Flask, jsonify, request
import joblib

# load model
model = joblib.load(open('model/DecisionTreeClassifier.model','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    scaler = model['imputer']
    le = model['labelencoder']
    classifier = model['model']
    version = model['scikit_version']
    print(version)

    numeric_col_names = ['amount_paid', 'approved_amount', 'disbursed_mail_flag', 'interest', 'latitude',
                         'loan_overdue_days', 'longitude'
        , 'penalty', 'previous_amount_payable', 'total_payable', 'loan_tenure', 'loan_amount', 'net_income']

    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)
    data_df['amount_paid'] = scaler.fit_transform(data_df['amount_paid'].values.reshape(-1, 1))
    data_df['approved_amount'] = scaler.fit_transform(data_df['approved_amount'].values.reshape(-1, 1))
    data_df['disbursed_mail_flag'] = scaler.fit_transform(data_df['disbursed_mail_flag'].values.reshape(-1, 1))
    data_df['interest'] = scaler.fit_transform(data_df['interest'].values.reshape(-1, 1))
    data_df['latitude'] = scaler.fit_transform(data_df['latitude'].values.reshape(-1, 1))
    data_df['loan_overdue_days'] = scaler.fit_transform(data_df['loan_overdue_days'].values.reshape(-1, 1))

    data_df['longitude'] = scaler.fit_transform(data_df['longitude'].values.reshape(-1, 1))
    data_df['penalty'] = scaler.fit_transform(data_df['penalty'].values.reshape(-1, 1))
    data_df['previous_amount_payable'] = scaler.fit_transform(data_df['previous_amount_payable'].values.reshape(-1, 1))
    data_df['total_payable'] = scaler.fit_transform(data_df['total_payable'].values.reshape(-1, 1))

    data_df['loan_tenure'] = scaler.fit_transform(data_df['loan_tenure'].values.reshape(-1, 1))
    data_df['loan_amount'] = scaler.fit_transform(data_df['loan_amount'].values.reshape(-1, 1))
    # data_df['outstanding_obligation_loan_amount'] = scaler.fit_transform(data_df['outstanding_obligation_loan_amount'].values.reshape(-1,1))
    data_df['net_income'] = scaler.fit_transform(data_df['net_income'].values.reshape(-1, 1))

    df1 = data_df[numeric_col_names]


    # predictions
    result = classifier.predict(df1)

    # send back to browser
    output = {'results': result[0]}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)

