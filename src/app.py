'''
 # @ Create Time: 2023-04-16 21:21:30.169847
'''
import base64
import datetime
import io
import requests, json
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import os
import pathlib
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__, title="myapp",external_stylesheets=[dbc.themes.BOOTSTRAP])

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server
#default_file_path = 'C:\Users\Owner\Desktop\cudash\data.csv'

def load_data(data_file: str) -> pd.DataFrame:
    '''
    Load data from /data directory
    '''
    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath("data").resolve()
    return pd.read_csv(DATA_PATH.joinpath(data_file))

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

# Sidebar content and style
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
                dbc.NavLink("Page 1", href="/page-1", id="page-1-link"),
                dbc.NavLink("Page 2", href="/page-2", id="page-2-link"),
                dbc.NavLink("Page 3", href="/page-3", id="page-3-link"),
                dbc.NavLink("Page 4", href="/page-4", id="page-4-link"),
                dbc.NavLink("Page 5", href="/page-5", id="page-5-link"),
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
    html.H3("Page 1", style={"textAlign": "center"}),
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
    html.H3("Page 2", style={"textAlign": "center"}),
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
            html.Div(dcc.Graph(id='line-plot-2'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginRight': '5px',
                            'marginBottom': '10px'})
        ], width=6),
        dbc.Col([
            html.Div(dcc.Graph(id='bar-plot-2'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginLeft': '5px',
                            'marginBottom': '10px'})
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='scatter-plot-2'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginRight': '5px'})
        ], width=6),
        dbc.Col([
            html.Div(dcc.Graph(id='weather-plot-2'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginLeft': '5px'})
        ], width=6),
    ])
])

page_3_layout = dbc.Container([
    html.H3("Page 3", style={"textAlign": "center"}),
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
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.DatePickerRange(
                id='date-picker-range-3',
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
            html.Div(dcc.Graph(id='line-plot-3'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginRight': '5px',
                            'marginBottom': '10px'})
        ], width=6),
        dbc.Col([
            html.Div(dcc.Graph(id='bar-plot-3'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginLeft': '5px',
                            'marginBottom': '10px'})
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='scatter-plot-3'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginRight': '5px'})
        ], width=6),
        dbc.Col([
            html.Div(dcc.Graph(id='weather-plot-3'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginLeft': '5px'})
        ], width=6),
    ])
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
            dcc.DatePickerRange(
                id='date-picker-range-4',
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
            html.Div(dcc.Graph(id='line-plot-4'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginRight': '5px',
                            'marginBottom': '10px'})
        ], width=6),
        dbc.Col([
            html.Div(dcc.Graph(id='bar-plot-4'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginLeft': '5px',
                            'marginBottom': '10px'})
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='scatter-plot-4'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginRight': '5px'})
        ], width=6),
        dbc.Col([
            html.Div(dcc.Graph(id='weather-plot-4'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginLeft': '5px'})
        ], width=6),
    ])
])

page_5_layout = dbc.Container([
    html.H3("Page 5", style={"textAlign": "center"}),
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
            html.Div(dcc.Graph(id='line-plot-5'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginRight': '5px',
                            'marginBottom': '10px'})
        ], width=6),
        dbc.Col([
            html.Div(dcc.Graph(id='bar-plot-5'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginLeft': '5px',
                            'marginBottom': '10px'})
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='scatter-plot-5'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginRight': '5px'})
        ], width=6),
        dbc.Col([
            html.Div(dcc.Graph(id='weather-plot-5'),
                     style={'border': '1px solid black', 'padding': '10px', 'marginLeft': '5px'})
        ], width=6),
    ])
])

def load_default_data():
    df = load_data('data.csv')
    # Convert "Date" column to datetime format
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

    # Convert "Time" column to datetime format
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time

    # Combine the "Date" and "Time" columns into a new "DateTime" column
    df["DateTime"] = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str))
    df['color'] = df['41CS.CH.PLANT.KW/TON KW/TON'].apply(lambda y: 'green' if y < 0.65 else ('blue' if y < 0.72 else 'red'))
    return df
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'xlsx' in filename:
            df = pd.read_csv(io.BytesIO(decoded), engine='c')
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


def update_weather_plot(start_date, end_date):
    if start_date is None or end_date is None:
        api_key = "36f887c81128fcde68b26f7334e939ab"
        city = "new%20york"
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"

        response = requests.get(url)
        data = response.json()
        response = requests.get(url)
        data = response.json()

        temp = round(data["main"]["temp"] - 273.15, 2)
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
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Temperature", "Humidity", "Wind Speed", "Weather Description"),    
                            specs=[[{"type": "domain"}, {"type": "domain"}],
                                [{"type": "domain"}, {"type": "domain"}]],)

        # Add traces for each subplot
        fig.add_trace(go.Indicator(mode="number", value=temp, title="Temperature (°C)"), row=1, col=1)
        fig.add_trace(go.Indicator(mode="number", value=humidity, title="Humidity (%)"), row=1, col=2)
        fig.add_trace(go.Indicator(mode="number", value=wind_speed, title="Wind Speed (m/s)"), row=2, col=1)
        fig.add_trace(go.Indicator(mode="number", value=weather_description_value, title="Weather Description"), row=2, col=2)

        fig.update_layout(height=600, title_text="Real-time Weather Data for New York")


        temp = round(data["main"]["temp"] - 273.15, 2)
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        weather_description = data["weather"][0]["description"]

        # Create figure with subplots
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Temperature", "Humidity", "Wind Speed", "Weather Description"),    
                            specs=[[{"type": "domain"}, {"type": "domain"}],
                                   [{"type": "domain"}, {"type": "domain"}]],)

        # Add traces for each subplot
        fig.add_trace(go.Indicator(mode="number", value=temp, title="Temperature (°C)"), row=1, col=1)
        fig.add_trace(go.Indicator(mode="number", value=humidity, title="Humidity (%)"), row=1, col=2)
        fig.add_trace(go.Indicator(mode="number", value=wind_speed, title="Wind Speed (m/s)"), row=2, col=1)
        fig.add_trace(go.Indicator(mode="number", value=weather_description_value, title="Weather Description"), row=2, col=2)

        fig.update_layout(height=600, title_text="Real-time Weather Data for New York")
    else:
        # Plot historical weather data
        df_weather = pd.read_csv("data/weather_data.csv")

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        df_weather = df_weather[(df_weather['dt_iso'] >= start_date) & (df_weather['dt_iso'] <= end_date)]

        fig = px.scatter(df_weather, x="dt_iso", y="temp", title="Temperature")

    return fig

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

    scatter_plot1 = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.KW/TON KW/TON', title='Line Plot',
                               color='41CS.CH.PLANT.KW/TON KW/TON',
                               color_continuous_scale=custom_color_scale,
                               range_color=(0, 0.72))
    scatter_plot1.update_xaxes(title_text='Date')
    scatter_plot1.update_yaxes(title_text='KW/TON')
    scatter_plot1.update_xaxes(title_text='Date')
    scatter_plot1.update_yaxes(title_text='KW/TON')

    scatter_plot2= px.scatter(df, x='DateTime', y='41CS.CH.PLANT.KW KW', title='Bar Plot')
    scatter_plot2.update_xaxes(title_text='Date')
    scatter_plot2.update_yaxes(title_text='KW')

    scatter_plot3 = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.TONS TONS', title='Scatter Plot')
    scatter_plot3.update_xaxes(title_text='Date')
    scatter_plot3.update_yaxes(title_text='TON')



    weather_plot = update_weather_plot(start_date, end_date)

    return scatter_plot1, scatter_plot2, scatter_plot3, weather_plot

@app.callback(
    Output('line-plot-2', 'figure'),
    Output('bar-plot-2', 'figure'),
    Output('scatter-plot-2', 'figure'),
    Output('efficiency-plot', 'figure'),
    Input('upload-data-2', 'contents'),
    Input('upload-data-2', 'filename'),
    Input('date-picker-range-2', 'start_date'),
    Input('date-picker-range-2', 'end_date')
)
def update_output_2(contents, filename, start_date, end_date):
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

    scatter_plot1 = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.KW/TON KW/TON', title='Line Plot',
                               color='41CS.CH.PLANT.KW/TON KW/TON',
                               color_continuous_scale=custom_color_scale,
                               range_color=(0, 0.72))
    scatter_plot1.update_xaxes(title_text='Date')
    scatter_plot1.update_yaxes(title_text='KW/TON')
    scatter_plot1.update_xaxes(title_text='Date')
    scatter_plot1.update_yaxes(title_text='KW/TON')
    scatter_plot2= px.scatter(df, x='DateTime', y='41CS.CH.PLANT.KW KW', title='Bar Plot')
    scatter_plot2.update_xaxes(title_text='Date')
    scatter_plot2.update_yaxes(title_text='KW')
    scatter_plot3 = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.TONS TONS', title='Scatter Plot')
    scatter_plot3.update_xaxes(title_text='Date')
    scatter_plot3.update_yaxes(title_text='TON')



    df['EWT'] = df['41CS.RF.BLRS:TS DEG F']
    df['LWT'] = df['41CS.RF.BLRS:TR DEG F']
    df['WBT'] = df['41CS.RF.BLRS:OAT DEG F']
    df['Range'] = df['EWT'] - df['LWT']
    df['Approach'] = df['LWT'] - df['WBT']
    df['Effectiveness'] = (df['Range'] * 100) / (df['EWT'] - df['WBT'])

    threshold = 5  # Replace this with the appropriate value
    df['Meets_Standard'] = df['Effectiveness'] >= threshold
    efficiency_plot = px.scatter(df, x='DateTime', y='Effectiveness', title='Cooling Tower Efficiency',
                                 color='Meets_Standard', color_discrete_sequence=['red', 'green'],
                                 labels={'Meets_Standard': 'ASHRAE Standard'})
    efficiency_plot.update_xaxes(title_text='Date')
    efficiency_plot.update_yaxes(title_text='Efficiency')

    return scatter_plot1, scatter_plot2, scatter_plot3, efficiency_plot

@app.callback(
    Output('line-plot-3', 'figure'),
    Output('bar-plot-3', 'figure'),
    Output('scatter-plot-3', 'figure'),
    Output('weather-plot-3', 'figure'),
    Input('upload-data-3', 'contents'),
    Input('upload-data-3', 'filename'),
    Input('date-picker-range-3', 'start_date'),
    Input('date-picker-range-3', 'end_date')
    )
def update_output_3(contents, filename, start_date, end_date):
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

    scatter_plot1 = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.KW/TON KW/TON', title='Line Plot',
                               color='41CS.CH.PLANT.KW/TON KW/TON',
                               color_continuous_scale=custom_color_scale,
                               range_color=(0, 0.72))
    scatter_plot1.update_xaxes(title_text='Date')
    scatter_plot1.update_yaxes(title_text='KW/TON')
    scatter_plot1.update_xaxes(title_text='Date')
    scatter_plot1.update_yaxes(title_text='KW/TON')
    scatter_plot2= px.scatter(df, x='DateTime', y='41CS.CH.PLANT.KW KW', title='Bar Plot')
    scatter_plot2.update_xaxes(title_text='Date')
    scatter_plot2.update_yaxes(title_text='KW')
    scatter_plot3 = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.TONS TONS', title='Scatter Plot')
    scatter_plot3.update_xaxes(title_text='Date')
    scatter_plot3.update_yaxes(title_text='TON')



    weather_plot = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.TONS TONS', title='Scatter Plot')
    weather_plot.update_xaxes(title_text='Date')
    weather_plot.update_yaxes(title_text='TON')

    return scatter_plot1, scatter_plot2, scatter_plot3, weather_plot

@app.callback(
    Output('line-plot-4', 'figure'),
    Output('bar-plot-4', 'figure'),
    Output('scatter-plot-4', 'figure'),
    Output('weather-plot-4', 'figure'),
    Input('upload-data-4', 'contents'),
    Input('upload-data-4', 'filename'),
    Input('date-picker-range-4', 'start_date'),
    Input('date-picker-range-4', 'end_date')
)
def update_output_4(contents, filename, start_date, end_date):
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

    scatter_plot1 = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.KW/TON KW/TON', title='Line Plot',
                               color='41CS.CH.PLANT.KW/TON KW/TON',
                               color_continuous_scale=custom_color_scale,
                               range_color=(0, 0.72))
    scatter_plot1.update_xaxes(title_text='Date')
    scatter_plot1.update_yaxes(title_text='KW/TON')
    scatter_plot1.update_xaxes(title_text='Date')
    scatter_plot1.update_yaxes(title_text='KW/TON')
    scatter_plot2= px.scatter(df, x='DateTime', y='41CS.CH.PLANT.KW KW', title='Bar Plot')
    scatter_plot2.update_xaxes(title_text='Date')
    scatter_plot2.update_yaxes(title_text='KW')
    scatter_plot3 = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.TONS TONS', title='Scatter Plot')
    scatter_plot3.update_xaxes(title_text='Date')
    scatter_plot3.update_yaxes(title_text='TON')

    weather_plot = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.TONS TONS', title='Scatter Plot')
    weather_plot.update_xaxes(title_text='Date')
    weather_plot.update_yaxes(title_text='TON')

    return scatter_plot1, scatter_plot2, scatter_plot3, weather_plot

@app.callback(
    Output('line-plot-5', 'figure'),
    Output('bar-plot-5', 'figure'),
    Output('scatter-plot-5', 'figure'),
    Output('weather-plot-5', 'figure'),
    Input('upload-data-5', 'contents'),
    Input('upload-data-5', 'filename'),
    Input('date-picker-range-5', 'start_date'),
    Input('date-picker-range-5', 'end_date')
)
def update_output_5(contents, filename, start_date, end_date):
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

    scatter_plot1 = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.KW/TON KW/TON', title='Line Plot',
                               color='41CS.CH.PLANT.KW/TON KW/TON',
                               color_continuous_scale=custom_color_scale,
                               range_color=(0, 0.72))
    scatter_plot1.update_xaxes(title_text='Date')
    scatter_plot1.update_yaxes(title_text='KW/TON')
    scatter_plot1.update_xaxes(title_text='Date')
    scatter_plot1.update_yaxes(title_text='KW/TON')
    scatter_plot2= px.scatter(df, x='DateTime', y='41CS.CH.PLANT.KW KW', title='Bar Plot')
    scatter_plot2.update_xaxes(title_text='Date')
    scatter_plot2.update_yaxes(title_text='KW')
    scatter_plot3 = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.TONS TONS', title='Scatter Plot')
    scatter_plot3.update_xaxes(title_text='Date')
    scatter_plot3.update_yaxes(title_text='TON')



    weather_plot = px.scatter(df, x='DateTime', y='41CS.CH.PLANT.TONS TONS', title='Scatter Plot')
    weather_plot.update_xaxes(title_text='Date')
    weather_plot.update_yaxes(title_text='TON')

    return scatter_plot1, scatter_plot2, scatter_plot3, weather_plot

if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)


