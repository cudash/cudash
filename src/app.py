'''
 # @ Create Time: 2023-04-16 21:21:30.169847
'''
import base64
import datetime
import io

import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State

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

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.Nav(
        [
            dbc.NavLink("Page 1", href="/page-1", id="page-1-link"),
            dbc.NavLink("Page 2", href="/page-2", id="page-2-link"),
        ],
        pills=True,
    ),
    html.Div(id='page-content')
])

##
page_1_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select an CSV File')]),
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
            dcc.Graph(id='line-plot'),
            dcc.Graph(id='bar-plot'),
            dcc.Graph(id='scatter-plot'),
            dcc.Graph(id='efficiency-plot')
        ])
    ])
])

page_2_layout = html.Div([])

# Callbacks and other functions go here





def load_default_data():
    df = load_data('data.csv')
    # Convert "Date" column to datetime format
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")

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
        if 'csv' in filename:
            df = pd.read_csv(io.BytesIO(decoded), engine='openpyxl')
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
    else:
        return page_1_layout  # Set the default page

@app.callback(
    Output('line-plot', 'figure'),
    Output('bar-plot', 'figure'),
    Output('scatter-plot', 'figure'),
    Output('efficiency-plot', 'figure'),  # Add this line
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


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)


