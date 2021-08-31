# -*- coding: utf-8 -*-
import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table

from datetime import date
import pickle

#import os
#os.chdir("./src")

import main
import model_performance as mp
#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title="Radek"

# Constants
PAGE_SIZE = 10

colors = {
    'background': '#222222',
    'text': '#d5d5d5'
}
# ----------------------------------------------------- 
# Import and clean data (importing csv into pandas)
#------------------------------------------------------
stockpicker_pnl = main.stockpicker_pnl.copy()
hodl_pnl = main.hodl_pnl.copy()
tickers = main.tickers
# Data for Graph and Table 
df = pd.concat(objs=[stockpicker_pnl, hodl_pnl], axis="columns")
equity = df[["Portfolio_Cumulative_Equity", "Buy_N_Hold_Cumulative_Equity"]]
equity = equity.stack().sort_index(level=1).reset_index() # stack to one series and sort strategies in a column
equity.columns = ["time", "strategy", "equity"]
# Initial Graph for Pnl of the Stockpicker Model and Hodl Strategy
fig_equity = px.line(
    data_frame=equity,
    x='time',
    y='equity',
    color='strategy',
    #title="Radek",
    template='plotly_dark'
)
# Initial Heatmap
features_contribution = mp.next_rebalance(model=main.best_mdl, X_test=main.predictors_out_bt)
fig_features_contrib = px.imshow(features_contribution,
                                labels=dict(x="Features", y="Stocks", color="Score"),
                                x=features_contribution.columns,
                                y=features_contribution.index,
                                template="plotly_dark",
                                width=800,
                                height=800
                                )
# ------------------------------------------------------------------------------
# App layout
# ------------------------------------------------------------------------------
app.layout = html.Div(style={'backgroundColor': colors["background"]}, children=[
    # Main Title
    html.H4("STOCKPICKER CONFIGURATOR", style={'text-align': 'left', "color": colors["text"]}),
    html.Br(),
    # Backtest Set-up
    # start date
    dcc.DatePickerSingle(
        id='start-date',
        min_date_allowed=date(2021, 1, 1),
        max_date_allowed=date.today(),
        initial_visible_month=date(2021, 1, 1),
        date=date(2021, 1, 1),
        style={'margin':'10px'} 
    ),
    # Number of Stocks
    dcc.Input(id="number_stocks", type="number", min=1, max=40, value=3, style={'margin':'10px'}), #value=3 because of 7 assets only
    # Leverage
    dcc.Input(id="leverage", type="number", min=1, max=2, step=0.1, value=1.3, style={'margin':'10px'}),
    # Start Cash
    dcc.Input(id="start_cash", type="number", min=40000, step=10000, value=100000, style={'margin':'10px'}),
    html.Button(id='backtest-button-state', n_clicks=0, children='Backtest', style={"color": colors["text"], 'margin':'10px'}), # backtest button
    # Graph for Pnl of the Stockpicker Model and Hodl Strategy
    dcc.Graph(id='stockpicker_line', figure=fig_equity),
    # Table with Pnl Results
    dash_table.DataTable(
        id="equity-table",
        columns=[
            {"name": i, "id": i} for i in equity.columns
        ],
        data=equity[::-1].head().to_dict("records"),
        page_current=0,
        page_size=PAGE_SIZE,
        page_action="custom",
        style_header={'backgroundColor': colors["background"], "color": colors["text"]},
        style_cell={'backgroundColor': colors["background"], "color": colors["text"]},
    ),
    html.Br(),
    html.H4("Next Rebalance", style={'text-align': 'left', "color": colors["text"]}),
    # Next Rebalance Stocks
    dcc.Checklist(
            id="all-or-none",
            options=[{"label": "Select All", "value": "All"}],
            value=["All"],
            labelStyle={"display": "inline-block", "color": colors["text"]},
        ),
    dcc.Checklist(
            id="desired_stocks",
            options=[{"label": s, "value": s} for s in features_contribution.index],
            value=[],
            labelStyle={"display": "block", "color": colors["text"]},
        ),
    # Heatmap for Features Contribution
    html.H6(children='Features Contribution', style={
        'color': colors["text"]
        }),
    dcc.Graph(id="feature_heatmap", figure=fig_features_contrib, style={
        "width": "80%"
    })
])

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
# @app.callback(
#     Output('equity-table', 'data'),
#     Input('equity-table', "page_current"),
#     Input('equity-table', "page_size"))
# def update_table(page_current,page_size):
#     return equity.iloc[
#         page_current*page_size:(page_current+ 1)*page_size
#     ].to_dict('records')

@app.callback(
    Output("desired_stocks", "value"),
    Input("all-or-none", "value"),
    State("desired_stocks", "options"),)

def select_all_none(all_selected, options):
    all_or_none = []
    all_or_none = [option["value"] for option in options if all_selected]
    return all_or_none
    
@app.callback(
    Output(component_id="stockpicker_line", component_property="figure"), # graph
    Output(component_id="equity-table", component_property="data"),
    Input(component_id="backtest-button-state", component_property="n_clicks"),
    State(component_id="start-date", component_property="date"),
    State(component_id="desired_stocks", component_property="value"),
    State(component_id="number_stocks", component_property="value"),
    State(component_id="leverage", component_property="value"),
    State(component_id="start_cash", component_property="value"))

def make_backtest(n_clicks, start_date, desired_stocks, number_assets, leverage, start_cash):
    # ctx = dash.callback_context
    # print("""""""""""""""""""""""""""""""")
    # print("CTX", ctx)
    # print("""""""""""""""""""""""""""""""")

    # Backtest
    stockpicker_pnl, hodl_pnl = mp.backtest_stockpicker(
        model=main.best_mdl,
        df=main.df,
        X_train=main.predictors_in,
        y_train=main.target_in,
        X_test=main.predictors_out_bt,
        y_test=main.target_out_bt,
        start_date=start_date,
        freq_rebalanc=main.freq_rebalanc,
        leverage=leverage,
        number_assets=number_assets,
        start_cash=start_cash,
        desired_stocks=desired_stocks
    )

    df = pd.concat(objs=[stockpicker_pnl, hodl_pnl], axis="columns")
    df = df.loc[start_date:, :]
    equity = df[["Portfolio_Cumulative_Equity", "Buy_N_Hold_Cumulative_Equity"]]
    equity = equity.stack().sort_index(level=1).reset_index() # stack to one series and sort strategies in a column
    equity.columns = ["time", "strategy", "equity"]

    # Graph for Pnl of the Stockpicker Model and Hodl Strategy
    fig_equity = px.line(
        data_frame=equity,
        x='time',
        y='equity',
        color='strategy',
        #title="Radek",
        template='plotly_dark'
    )
   
   # Paging
    return fig_equity, equity[::-1].head().to_dict("records")
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    #gunicorn -b 0.0.0.0:8050 app:app.server
    app.run_server(port=8050, host="127.0.0.1")
