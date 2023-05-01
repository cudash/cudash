from dash.exceptions import PreventUpdate
import base64
import datetime
from dash import dash_table
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from plotly.subplots import make_subplots
import io
import plotly.graph_objs as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State, MATCH
import requests
import pandas as pd
import plotly.express as px
import numpy as np
import pathlib


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#For spaces classified as occupancy groups E and I-4: multiply the building emissions intensity limit of (2024) 0.01074 tCO2e/sf (2030) 0.0042 tCO2e/sf by the corresponding gross floor area (sf);
def load_data(data_file: str) -> pd.DataFrame:
    '''
    Load data from /data directory
    '''
    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath("data").resolve()
    return pd.read_excel(DATA_PATH.joinpath(data_file), engine='openpyxl')

def load_datac(data_file: str) -> pd.DataFrame:
    '''
    Load data from /data directory
    '''
    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath("data").resolve()
    return pd.read_csv(DATA_PATH.joinpath(data_file))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Suppress callback exceptions

app.config.suppress_callback_exceptions = True
def get_weather_data(start_date, end_date):
    url = f'https://api.openweathermap.org/data/2.5/weather?q=New+York&appid=YOUR_API_KEY&start={start_date}&end={end_date}'
    response = requests.get(url)
    data = response.json()
    return data

def parse_weather_data(data):
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    return temperature, humidity

def load_default_data():
    df = load_data('41CSDATA.xlsx')
    # Convert "Date" column to datetime format
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")

    # Convert "Time" column to datetime format
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time

    # Combine the "Date" and "Time" columns into a new "DateTime" column
    df["DateTime"] = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str))
    df['color'] = df['41CS.CH.PLANT.KW/TON KW/TON'].apply(lambda y: 'green' if y < 0.65 else ('blue' if y < 0.72 else 'red'))
    return df

def load_default_data2():
    df = load_data('CarbonTrackingFile.xlsx')
    # Convert "Date" column to datetime format
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return df

df2 = load_default_data2()
df2 = df2.dropna(subset=['equipment room', 'equipment type'])

def load_default_data3():
    df = load_data('CWST.xlsx')
    return df

def load_default_datac4():
    df = load_datac('cudash.csv')
    return df
# Suppress callback exceptions

app.config.suppress_callback_exceptions = True
def get_weather_data(start_date, end_date):
    url = f'https://api.openweathermap.org/data/2.5/weather?q=New+York&appid=YOUR_API_KEY&start={start_date}&end={end_date}'
    response = requests.get(url)
    data = response.json()
    return data

def parse_weather_data(data):
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    return temperature, humidity

SIDEBAR_STYLE = {
    "top": 0,
    "right": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "z-index": 1,
    "transition": "all 0.3s",
    "border-left": "1px solid #dee2e6",
}

sidebar = html.Div(
    [
        html.H2("Dash App", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Chiller Plant", href="/page-1", id="page-1-link"),
                dbc.NavLink("Air Handler", href="/page-2", id="page-2-link"),
                dbc.NavLink("Cooling Tower/CWST", href="/page-3", id="page-3-link"),
                dbc.NavLink("Equipment", href="/page-4", id="page-4-link"),
                dbc.NavLink("Carbon Tracking", href="/page-5", id="page-5-link"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
    id="sidebar",
)


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        sidebar,
        html.Div(
            [
                html.Div(id="page-content", style={"padding": "2rem 1rem"})
            ],
            style={"margin-right": "0rem", "flex-grow":"2"},
        ),
    ], style={"display":"flex"}
)

page_1_layout = dbc.Container([
    html.H3("Chiller Plant", style={"textAlign": "center"}),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select an Excel File')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                accept='.csv'
            )
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date_placeholder_text='Start Date',
                end_date_placeholder_text='End Date',
                display_format='YYYY-MM-DD',
                min_date_allowed=datetime.datetime(1970, 1, 1),
                max_date_allowed=datetime.datetime.now(),
                initial_visible_month=datetime.datetime.now()
            )
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='line-plot'), style={'border': '1px solid black', 'padding': '10px', 'marginRight': '5px', 'marginBottom': '10px'})
        ], width=6),
        dbc.Col([
            html.Div(dcc.Graph(id='bar-plot'), style={'border': '1px solid black', 'padding': '10px', 'marginLeft': '5px', 'marginBottom': '10px'})
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='scatter-plot'), style={'border': '1px solid black', 'padding': '10px', 'marginRight': '5px'})
        ], width=6),
        dbc.Col([
            html.Div(dcc.Graph(id='weather-plot'), style={'border': '1px solid black', 'padding': '10px', 'marginLeft': '5px'})
        ], width=6),
    ])
])



page_2_layout = dbc.Container([
    html.H3("Air Handling Units", style={"textAlign": "center"}),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data-2',
                children=html.Div(['Drag and Drop or ', html.A('Select an Excel File')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                accept='.csv'
            )
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.DatePickerRange(
                id='date-picker-range-2',
                start_date_placeholder_text='Start Date',
                end_date_placeholder_text='End Date',
                display_format='YYYY-MM-DD',
                min_date_allowed=datetime.datetime(1970, 1, 1),
                max_date_allowed=datetime.datetime.now(),
                initial_visible_month=datetime.datetime.now()
            )
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='ah01-sup-flow-plot'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginBottom': '10px'})
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='ah02-sup-flow-plot'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginBottom': '10px'})
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='ah03-sup-flow-plot'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginBottom': '10px'})
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='ah04-sup-flow-plot'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginBottom': '10px'})
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='ah05-sup-flow-plot'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginBottom': '10px'})
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='ah06-sup-flow-plot'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginBottom': '10px'})
        ], width=12),
    ])
])

page_3_layout = html.Div([
    html.H1("Optimizing Chiller Plant"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Inputs"),
                dbc.CardBody([
                    dbc.Label("Load (tons)"),
                    dbc.Input(id='load_tons', type="number", min=0, step=0.1, placeholder="Enter load (tons)"),
                    dbc.Label("Wet Bulb (°F)"),
                    dbc.Input(id='wet_bulb', type="number", min=0, step=0.1, placeholder="Enter wet bulb (°F)"),
                    dbc.Label("Current CWST"),
                    dbc.Input(id='current_cwst', type="number", min=0, step=0.1, placeholder="Enter current CWST (°F)"),
                    dbc.Label("Current Plant kW"),
                    dbc.Input(id='current_plant_kw', type="number", min=0, step=0.1, placeholder="Enter current plant kW"),
                    dbc.Button("Calculate", id="calculate_btn", color="primary", className="mt-2")
                ]),
            ]),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Optimal CWST (°F)"),
                dbc.CardBody([
                    html.Div(id='optimal_cwst')
                ]),
            ]),
            dbc.Card([
                dbc.CardHeader("Energy Use Difference (kW):"),
                dbc.CardBody([
                    html.Div(id='energy_savings')
                ]),
            ]),
        ], width=8),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data-3',
                children=html.Div(['Drag and Drop or ', html.A('Select an Excel File')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                accept='.csv'
            )
        ]),
        dbc.Col([
            html.Div([
                dbc.Label(""),
                dcc.DatePickerRange(
                    id='date_range_picker_3',
                    start_date_placeholder_text="Start Date",
                    end_date_placeholder_text="End Date",
                    min_date_allowed=datetime.date(2020, 1, 1),
                    max_date_allowed=datetime.date.today(),
                    initial_visible_month=datetime.date.today(),
                ),
            ], style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '20px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'padding': '5px'
            }),
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='CoolingTower1'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginBottom': '10px'})
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='CoolingTower2'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginBottom': '10px'})
        ], width=12),
    ]),
])




page_4_layout = dbc.Container([
    html.H3("Page 4", style={"textAlign": "center"}),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data-4',
                children=html.Div(['Drag and Drop or ', html.A('Select an Excel File')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                accept='.csv'
            )
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(id='needs_attention_list', children=[
                html.H4("Equipment Log"),
                dash_table.DataTable(
                    id='broken_dampers_table',
                    columns=[
                        {"name": "equipment room", "id": "equipment room"},
                        {"name": "equipment type", "id": "equipment type"},
                        {"name": "notes", "id": "notes", "editable": True},
                        {"name": "work done", "id": "work done", "editable": True},
                    ],
                    data=df2[["equipment room", "equipment type"]].assign(notes="", work_done="").to_dict('records'),
                    editable=True,
                    row_deletable=True,
                ),
                dbc.Button('Add Item', id='add_item_button', color='primary', size='sm', style={'margin-top': '10px'})
            ])
        ], width=12),
    ]),
])



page_5_layout = dbc.Container([
    html.H3("Carbon Tracking", style={"textAlign": "center"}),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data-5',
                children=html.Div(['Drag and Drop or ', html.A('Select an Excel File')]),
                style={
                   'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                accept='.csv'
            )
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.DatePickerRange(
                id='date-picker-range-5',
                start_date_placeholder_text='Start Date',
                end_date_placeholder_text='End Date',
                display_format='YYYY-MM-DD',
                min_date_allowed=datetime.datetime(1970, 1, 1),
                max_date_allowed=datetime.datetime.now(),
                initial_visible_month=datetime.datetime.now()
            )
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='pie_chart'),
                     style={'border': '1px solid black', 'padding': '10px', 'margin': '10px'})
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='co2_plot'),
                     style={'border': '1px solid black', 'padding': '10px', 'margin': '10px'})
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='yearly_energy_plot'),
                     style={'border': '1px solid black', 'padding': '10px', 'margin': '10px'})
        ], width=12),
    ]),
    # dbc.Row([
    #     dbc.Col([
    #         dbc.Input(id='new_item', placeholder='Add new item', type='text'),
    #         dbc.Button('Add', id='add_item_button', color='primary', n_clicks=0),
    #     ], width=12),
    # ]),
])

def load_default_data4():
    df = load_default_datac4()
    # Convert "Date" column to datetime format
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")

    # Convert "Time" column to datetime format
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time

    # Combine the "Date" and "Time" columns into a new "DateTime" column
    df["DateTime"] = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str))
    df['color'] = df['41CS.CH.PLANT.KW/TON KW/TON'].apply(lambda y: 'green' if y < 0.65 else ('blue' if y < 0.72 else 'red'))
    return df

def has_columns(df, required_columns):
    return all(column in df.columns for column in required_columns)

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'xlsx' in filename:
            df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')
            # Convert "Date" column to datetime format
            df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")

            # Convert "Time" column to datetime format
            df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time

            # Combine the "Date" and "Time" columns into a new "DateTime" column
            df["DateTime"] = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str))
            df['color'] = df['41CS.CH.PLANT.KW/TON KW/TON'].apply(lambda y: 'green' if y < 0.65 else ('blue' if y < 0.72 else 'red'))
            return df

    except Exception as e:
        print(e)
        return None

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    elif pathname == '/page-3':
        return page_3_layout
    elif pathname == '/page-4':
        return page_4_layout
    elif pathname == '/page-5':
        return page_5_layout
    else:
        return page_1_layout  # Set the default page


    url = f'https://api.openweathermap.org/data/2.5/weather?q=New+York&appid=YOUR_API_KEY&start={start_date}&end={end_date}'
    response = requests.get(url)
    data = response.json()
    return data

    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    return temperature, humidity

@app.callback(
    Output('line-plot', 'figure'),
    Output('bar-plot', 'figure'),
    Output('scatter-plot', 'figure'),
    Output('weather-plot', 'figure'),
    Input('upload-data', 'contents'),
    Input('upload-data', 'filename'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_output(contents, filename, start_date, end_date):
    if contents is None:
        df = load_default_data()
    else:
        df = parse_contents(contents, filename)
        if df is None:
            df = load_default_data()

    if start_date is not None and end_date is not None:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]


    custom_color_scale = [
        (0.0, 'green'),
        (0.64 / 0.72, 'green'),
        (0.64 / 0.72, 'blue'),
        (0.71 / 0.72, 'blue'),
        (0.71 / 0.72, 'red'),
        (1.0, 'red')
    ]

    scatter_plot1, scatter_plot2, scatter_plot3 = None, None, None

    if has_columns(df, ['DateTime', '41CS.CH.PLANT.KW/TON KW/TON']):
        scatter_plot1 = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.KW/TON KW/TON', title='KW/Ton',
                                   color='41CS.CH.PLANT.KW/TON KW/TON',
                                   color_continuous_scale=custom_color_scale,
                                   range_color=(0, 0.72))
        scatter_plot1.update_xaxes(title_text='Date')
        scatter_plot1.update_yaxes(title_text='KW/TON')

    if has_columns(df, ['DateTime', '41CS.CH.PLANT.KW KW']):
        scatter_plot2 = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.KW KW', title='KW')
        scatter_plot2.update_xaxes(title_text='Date')
        scatter_plot2.update_yaxes(title_text='KW')

    if has_columns(df, ['DateTime', '41CS.CH.PLANT.TONS TONS']):
        scatter_plot3 = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.TONS TONS', title='Ton')
        scatter_plot3.update_xaxes(title_text='Date')
        scatter_plot3.update_yaxes(title_text='TON')

    def update_weather_plot(start_date, end_date):
        if start_date is None or end_date is None:
            api_key = "36f887c81128fcde68b26f7334e939ab"
            city = "new%20york"
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"

            response = requests.get(url)
            data = response.json()
            response = requests.get(url)
            data = response.json()

            temp = round(((data["main"]["temp"] - 273.15)*1.8)+32, 2)
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
            weather_description = data["weather"][0]["description"]

            # Create a dictionary to map weather description values to numeric values
            weather_dict = {"clear sky": 1, "few clouds": 2, "scattered clouds": 3, "broken clouds": 4,
                            "overcast clouds": 5, "light rain": 6, "moderate rain": 7, "heavy intensity rain": 8}

            # Map the weather description value to a corresponding numeric value
            if weather_description in weather_dict:
                weather_description_value = weather_dict[weather_description]
            else:
                weather_description_value = 0

            # Create figure with subplots
            fig = make_subplots(rows=2, cols=2,
                                subplot_titles=("Temperature", "Humidity", "Wind Speed", "Weather Description"),
                                specs=[[{"type": "domain"}, {"type": "domain"}],
                                       [{"type": "domain"}, {"type": "domain"}]], )

            # Add traces for each subplot
            fig.add_trace(go.Indicator(mode="number", value=temp, title="Temperature (°C)"), row=1, col=1)
            fig.add_trace(go.Indicator(mode="number", value=humidity, title="Humidity (%)"), row=1, col=2)
            fig.add_trace(go.Indicator(mode="number", value=wind_speed, title="Wind Speed (m/s)"), row=2, col=1)
            fig.add_trace(go.Indicator(mode="number", value=weather_description_value, title="Weather Description"),
                          row=2, col=2)

            fig.update_layout(height=600, title_text="Real-time Weather Data for New York")

            temp = round(data["main"]["temp"] - 273.15, 2)
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
            weather_description = data["weather"][0]["description"]

            # Create figure with subplots
            fig = make_subplots(rows=2, cols=2,
                                subplot_titles=("Temperature", "Humidity", "Wind Speed", "Weather Description"),
                                specs=[[{"type": "domain"}, {"type": "domain"}],
                                       [{"type": "domain"}, {"type": "domain"}]], )

            # Add traces for each subplot
            fig.add_trace(go.Indicator(mode="number", value=temp, title="Temperature (°F)"), row=1, col=1)
            fig.add_trace(go.Indicator(mode="number", value=humidity, title="Humidity (%)"), row=1, col=2)
            fig.add_trace(go.Indicator(mode="number", value=wind_speed, title="Wind Speed (m/s)"), row=2, col=1)
            fig.add_trace(go.Indicator(mode="number", value=weather_description_value, title="Weather Description"),
                          row=2, col=2)

            fig.update_layout(height=600, title_text="Real-time Weather Data for New York")

            return fig

    weather_plot = update_weather_plot(None, None)
    return scatter_plot1, scatter_plot2, scatter_plot3, weather_plot

@app.callback(
    Output('ah01-sup-flow-plot', 'figure'),
    Output('ah02-sup-flow-plot', 'figure'),
    Output('ah03-sup-flow-plot', 'figure'),
    Output('ah04-sup-flow-plot', 'figure'),
    Output('ah05-sup-flow-plot', 'figure'),
    Output('ah06-sup-flow-plot', 'figure'),
    Input('upload-data-2', 'contents'),
    Input('upload-data-2', 'filename'),
    Input('date-picker-range-2', 'start_date'),
    Input('date-picker-range-2', 'end_date')
)
def update_output_2(contents, filename, start_date, end_date):
    if contents is None:
        df = load_default_data4()
    else:
        df = parse_contents(contents, filename)
        if df is None:
            df = load_default_data4()

    if start_date is None or end_date is None:
        end_date = df['Date'].max()
        start_date = end_date - pd.DateOffset(months=1)

    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    import plotly.express as px

    def create_plot(column_name, title):
        if column_name in df.columns:
            plot = px.scatter(df, x='DateTime', y=column_name, title=title)
            plot.update_xaxes(title_text='Date')
            plot.update_yaxes(title_text= "Airflow (CFM)")
        else:
            plot = None
        return plot

    ah01_sup_flow_plot = create_plot('41CS.LL2.AH01.SUP.FLOW CFM', 'AH01 SUPPLY FLOW')
    ah02_sup_flow_plot = create_plot('41CS.LL2.AH02.SUP.FLOW CFM', 'AH02 SUPPLY FLOW')
    ah03_sup_flow_plot = create_plot('41CS.LL2.AH03.SUP.FLOW CFM', 'AH03 SUPPLY FLOW')
    ah04_sup_flow_plot = create_plot('41CS.RF.AH04.SUP.FLOW CFM', 'AH04 SUPPLY FLOW')
    ah05_sup_flow_plot = create_plot('41CS.RF.AH05.SUP.FLOW CFM', 'AH05 SUPPLY FLOW')
    ah06_sup_flow_plot = create_plot('41CS.RF.AH06.SUP.FLOW CFM', 'AH06 SUPPLY FLOW')

    return (ah01_sup_flow_plot, ah02_sup_flow_plot, ah03_sup_flow_plot,
            ah04_sup_flow_plot, ah05_sup_flow_plot, ah06_sup_flow_plot)


@app.callback(
    [Output("optimal_cwst", "children"),
     Output("energy_savings", "children")],
    [Input("calculate_btn", "n_clicks")],
    [State("load_tons", "value"),
     State("wet_bulb", "value"),
     State("current_cwst", "value"),
     State("current_plant_kw", "value"),
     ]
)
def calculate_optimal_cwst_and_savings(n_clicks, chiller_load, wet_bulb, cwst_setpoint, current_plant_kw):
    if n_clicks is None or chiller_load is None or wet_bulb is None or cwst_setpoint is None or current_plant_kw is None:
        raise PreventUpdate

    data = load_default_data3()
    x_values = ['Chiller Loads', 'Wet Bulb', 'CWST Setpoint']
    y_values = ['Total Plant kW']
    degree = 2

    x_dataPoly = np.array(data[x_values])
    y_dataPoly = np.array(data[y_values])
    x_trainP, x_testP, y_trainP, y_testP = train_test_split(x_dataPoly, y_dataPoly, test_size=0.2, random_state=125)

    poly = PolynomialFeatures(degree=degree)
    x_train_transformed = poly.fit_transform(x_trainP)
    x_test_transformed = poly.fit_transform(x_testP)

    lmPoly = LinearRegression()
    lmPoly.fit(x_train_transformed, y_trainP)

    # Calculate the optimal w
    optimal_w = (0.0256 * chiller_load) + (0.7693 * wet_bulb) + 20.23
    optimal_w = round(optimal_w)

    # Limit optimal_w between 68 and 78
    if optimal_w < 68:
        optimal_w = 68
    elif optimal_w > 78:
        optimal_w = 78

    # Predict the new plant kW
    predicted_kw = lmPoly.predict(poly.fit_transform(np.array([[chiller_load, wet_bulb, optimal_w]])))[0][0]

    # Calculate energy savings
    energy_savings = current_plant_kw - predicted_kw
    energy_savings = round(energy_savings)

    return optimal_w, energy_savings


@app.callback(
    Output('CoolingTower1', 'figure'),
    Output('CoolingTower2', 'figure'),
    Input('upload-data-3', 'contents'),
    Input('upload-data-3', 'filename'),
    Input('date_range_picker_3', 'start_date'),
    Input('date_range_picker_3', 'end_date')
)
def update_output_3(contents, filename, start_date, end_date):
    if contents is None:
        df = load_default_data4()
    else:
        df = parse_contents(contents, filename)
        if df is None:
            df = load_default_data4()

    if start_date is None or end_date is None:
        end_date = df['Date'].max()
        start_date = end_date - pd.DateOffset(months=1)

    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    import plotly.express as px

    def create_plot(column_name, title):
        if column_name in df.columns:
            plot = px.scatter(df, x='DateTime', y=column_name, title=title)
            plot.update_xaxes(title_text='Date')
            plot.update_yaxes(title_text= "Airflow (CFM)")
        else:
            plot = None
        return plot

    CoolingTower1 = create_plot('41CS.RF.CT01.KW KW', 'Cooling Tower 1 Energy Use (kW)')
    CoolingTower2 = create_plot('41CS.RF.CT02.KW KW', 'Cooling Tower 2 Energy Use (kW)')


    return (CoolingTower1, CoolingTower2)



@app.callback(
    Output('broken_dampers_table', 'data'),
    Input('add_item_button', 'n_clicks'),
    State('broken_dampers_table', 'data')
)
def add_item(n_clicks, rows):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    rows.append({c: '' for c in df2.columns})
    return rows


@app.callback(
    [Output('pie_chart', 'figure'),
     Output('yearly_energy_plot', 'figure'),
     Output('co2_plot', 'figure')],
    [Input('upload-data-5', 'contents'),
     Input('upload-data-5', 'filename'),
     Input('date-picker-range-5', 'start_date'),
     Input('date-picker-range-5', 'end_date')]
)
def update_output_5(contents, filename, start_date, end_date):
    if contents is None:
        df = load_default_data2()
    else:
        df = parse_contents(contents, filename)
        if df is None:
            df = load_default_data2()


    # Make sure the Date column is JSON serializable
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M').astype(str)

    if start_date is not None and end_date is not None:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Convert electricity use from kWh to MMBTU
    kwh_to_mmbtu = 0.003412
    df['41 CS Elec Use (MMBTU)'] = df['41 CS Elec Use (Kwh)'] * kwh_to_mmbtu
    df['FB Elec Use (MMBTU)'] = df['FB Elec Use (Kwh)'] * kwh_to_mmbtu
    df['RH Elec Use (MMBTU)'] = df['RH Elec Use (Kwh)'] * kwh_to_mmbtu

    # Convert gas use from therms to MMBTU
    therms_to_mmbtu = 0.1
    df['41 CS Gas Use (MMBTU)'] = df['41 CS Gas Use (therm)'] * therms_to_mmbtu
    df['FB Gas Use (MMBTU)'] = df['FB Gas Use (therm)'] * therms_to_mmbtu
    df['RH Gas Use (MMBTU)'] = df['RH Gas Use (therm)'] * therms_to_mmbtu

    # Calculate total energy use in MMBTU
    df['Total Energy Use (MMBTU)'] = df['41 CS Elec Use (MMBTU)'] + df['FB Elec Use (MMBTU)'] + df['RH Elec Use (MMBTU)'] + df['41 CS Gas Use (MMBTU)'] + df['FB Gas Use (MMBTU)'] + df['RH Gas Use (MMBTU)']

    # Group by month and aggregate the energy usage
    energy_data = df.groupby(['Month']).agg({'41 CS Elec Use (MMBTU)': 'sum', 'FB Elec Use (MMBTU)': 'sum', 'RH Elec Use (MMBTU)': 'sum', '41 CS Gas Use (MMBTU)': 'sum', 'FB Gas Use (MMBTU)': 'sum', 'RH Gas Use (MMBTU)': 'sum', 'Total Energy Use (MMBTU)': 'sum'}).reset_index()
    energy_data['41 CS Total Energy'] = energy_data['41 CS Elec Use (MMBTU)'] + energy_data['41 CS Gas Use (MMBTU)']
    energy_data['FB Total Energy'] = energy_data['FB Elec Use (MMBTU)'] + energy_data['FB Gas Use (MMBTU)']
    energy_data['RH Total Energy'] = energy_data['RH Elec Use (MMBTU)'] + energy_data['RH Gas Use (MMBTU)']

    pie_data = []
    for index, row in energy_data.iterrows():
        pie_data.append(
            {'Month': row['Month'], 'Building': '41 CS - Electricity', 'Value': row['41 CS Elec Use (MMBTU)']})
        pie_data.append(
            {'Month': row['Month'], 'Building': '41 CS - Natural Gas', 'Value': row['41 CS Gas Use (MMBTU)']})
        pie_data.append({'Month': row['Month'], 'Building': 'FB - Electricity', 'Value': row['FB Elec Use (MMBTU)']})
        pie_data.append({'Month': row['Month'], 'Building': 'FB - Natural Gas', 'Value': row['FB Gas Use (MMBTU)']})
        pie_data.append({'Month': row['Month'], 'Building': 'RH - Electricity', 'Value': row['RH Elec Use (MMBTU)']})
        pie_data.append({'Month': row['Month'], 'Building': 'RH - Natural Gas', 'Value': row['RH Gas Use (MMBTU)']})

    pie_df = pd.DataFrame(pie_data)

    color_map = {
        '41 CS - Electricity (MMBtu)': 'blue',
        '41 CS - Natural Gas (MMBtu)': 'green',
        'FB - Electricity (MMBtu)': 'red',
        'FB - Natural Gas (MMBtu)': 'purple',
        'RH - Electricity (MMBtu)': 'orange',
        'RH - Natural Gas (MMBtu)': 'brown',
    }

    # Create the pie chart
    pie_chart = px.pie(pie_df, height=900,values='Value', names='Building', color='Building', color_discrete_map=color_map,
                       title='Total Energy Use in MMBTU by Building and Energy Type')

    # Get the min and max month in the dataset
    min_month = df['Date'].min()
    max_month = df['Date'].max()

    # Add the date range to the title
    pie_chart.update_layout(
        title=f'Total Energy Use in MMBTU by Building and Energy Type<br>{min_month} to {max_month}')

    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    yearly_energy_data = df.groupby(['Year']).agg(
        {'41 CS Elec Use (MMBTU)': 'sum', 'FB Elec Use (MMBTU)': 'sum', 'RH Elec Use (MMBTU)': 'sum',
         '41 CS Gas Use (MMBTU)': 'sum', 'FB Gas Use (MMBTU)': 'sum', 'RH Gas Use (MMBTU)': 'sum'}).reset_index()
    yearly_energy_plot = go.Figure()

    yearly_energy_plot.add_trace(go.Bar(
        x=yearly_energy_data['Year'],
        y=yearly_energy_data['41 CS Elec Use (MMBTU)'],
        name='41 CS - Electricity',
        marker_color='blue'
    ))

    yearly_energy_plot.add_trace(go.Bar(
        x=yearly_energy_data['Year'],
        y=yearly_energy_data['41 CS Gas Use (MMBTU)'],
        name='41 CS - Natural Gas',
        marker_color='green'
    ))

    yearly_energy_plot.add_trace(go.Bar(
        x=yearly_energy_data['Year'],
        y=yearly_energy_data['FB Elec Use (MMBTU)'],
        name='FB - Electricity',
        marker_color='red'
    ))

    yearly_energy_plot.add_trace(go.Bar(
        x=yearly_energy_data['Year'],
        y=yearly_energy_data['FB Gas Use (MMBTU)'],
        name='FB - Natural Gas',
        marker_color='purple'
    ))

    yearly_energy_plot.add_trace(go.Bar(
        x=yearly_energy_data['Year'],
        y=yearly_energy_data['RH Elec Use (MMBTU)'],
        name='RH - Electricity',
        marker_color='orange'
    ))

    yearly_energy_plot.add_trace(go.Bar(
        x=yearly_energy_data['Year'],
        y=yearly_energy_data['RH Gas Use (MMBTU)'],
        name='RH - Natural Gas',
        marker_color='brown'
    ))

    yearly_energy_plot.update_layout(barmode='group', height=900,title='Yearly Energy Use by Building and Energy Type',
                                     xaxis_title='Year', yaxis_title='Energy Use (MMBTU)')

    natural_gas_coefficient = 0.05311  # metric tons CO2 per MMBTU
    electricity_coefficient = 0.098  # metric tons CO2 per MMBTU (converted from 0.000288 metric tons CO2 per kWh)

    yearly_energy_data['41 CS CO2 Emissions (Natural Gas)'] = yearly_energy_data[
                                                                  '41 CS Gas Use (MMBTU)'] * natural_gas_coefficient
    yearly_energy_data['41 CS CO2 Emissions (Electricity)'] = yearly_energy_data[
                                                                  '41 CS Elec Use (MMBTU)'] * electricity_coefficient

    yearly_energy_data['FB CO2 Emissions (Natural Gas)'] = yearly_energy_data[
                                                               'FB Gas Use (MMBTU)'] * natural_gas_coefficient
    yearly_energy_data['FB CO2 Emissions (Electricity)'] = yearly_energy_data[
                                                               'FB Elec Use (MMBTU)'] * electricity_coefficient

    yearly_energy_data['RH CO2 Emissions (Natural Gas)'] = yearly_energy_data[
                                                               'RH Gas Use (MMBTU)'] * natural_gas_coefficient
    yearly_energy_data['RH CO2 Emissions (Electricity)'] = yearly_energy_data[
                                                               'RH Elec Use (MMBTU)'] * electricity_coefficient

    ll97_limits = {
        '41 CS': 1933,  # Replace with the actual LL97 limit for 41 CS
        'FB': 1858,  # Replace with the actual LL97 limit for FB
        'RH': 644  # Replace with the actual LL97 limit for RH
    }

    co2_emissions_plot = go.Figure()

    # Adding traces for each building
    building_traces = [
        ('41 CS', 'green', 'blue'),
        ('FB', 'grey', 'red'),
        ('RH', 'brown', 'orange')
    ]

    for building, gas_color, elec_color in building_traces:
        co2_emissions_plot.add_trace(go.Bar(
            x=yearly_energy_data['Year'],
            y=yearly_energy_data[f'{building} CO2 Emissions (Natural Gas)'],
            name=f'{building} - Natural Gas',
            marker_color=gas_color,
            legendgroup=building
        ))

        co2_emissions_plot.add_trace(go.Bar(
            x=yearly_energy_data['Year'],
            y=yearly_energy_data[f'{building} CO2 Emissions (Electricity)'],
            name=f'{building} - Electricity',
            marker_color=elec_color,
            legendgroup=building,
        ))

    # Update layout for stacked bars and add legend
    co2_emissions_plot.update_layout(
        height=900,
        barmode='stack',
        title='Yearly CO2 Emissions by Building and Energy Type',
        xaxis_title='Year',
        yaxis_title='CO2 Emissions (Metric Tons)',
        legend=dict(orientation='h', yanchor='bottom', xanchor='right', y=1.02, x=1)
    )

    # Add lines for LL97 limits
    ll97_limits.update({'Campus': 4435})  # Replace 1800 with the actual LL97 limit for the campus

    colors = {'41 CS': 'brown', 'FB': 'red', 'RH': 'blue', 'Campus': 'purple'}

    for building, limit in ll97_limits.items():
        co2_emissions_plot.add_trace(go.Scatter(
            x=[yearly_energy_data['Year'].min(), yearly_energy_data['Year'].max()],
            y=[limit, limit],
            mode='lines',
            line=dict(color=colors[building], width=3),
            name=f'{building} LL97 Limit (Yearly tCO2e: {limit})',
            legendgroup=building,
            text=[f'{building} LL97 Limit'] * 2,
            hoverinfo='text'
        ))

    co2_emissions_plot.update_layout(
        # ... other layout settings ...
        legend=dict(
            x=0.5,  # Adjust the x position of the legend
            y=-0.1,  # Adjust the y position of the legend to place it below the graph
            xanchor='center',  # Anchor the legend at the center horizontally
            yanchor='top',  # Anchor the legend at the top vertically
            bgcolor='rgba(255, 255, 255, 0.5)'  # Set the background color of the legend
        )
    )

    return pie_chart, yearly_energy_plot, co2_emissions_plot

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)


