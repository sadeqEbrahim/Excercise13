from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Collect data from form
        type_of_renewable = request.form['Type_of_Renewable_Energy']
        installed_capacity = request.form['Installed_Capacity_MW']
        energy_production = request.form['Energy_Production_MWh']
        energy_consumption = request.form['Energy_Consumption_MWh']
        energy_storage = request.form['Energy_Storage_Capacity_MWh']

        # For now, just display the collected data
        return f"""
        <h1>Submitted Data</h1>
        <p>Type of Renewable Energy: {type_of_renewable}</p>
        <p>Installed Capacity (MW): {installed_capacity}</p>
        <p>Energy Production (MWh): {energy_production}</p>
        <p>Energy Consumption (MWh): {energy_consumption}</p>
        <p>Energy Storage Capacity (MWh): {energy_storage}</p>
        """


if __name__ == '__main__':
    app.run(host='0.0.0.0',
port=8080, debug=Truedocker pstats)