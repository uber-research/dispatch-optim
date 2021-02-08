import dash
import dash_core_components as dcc
import dash_html_components as html
#import dash_table_experiments as dt
from callbacks import *


app = dash.Dash(__name__, assets_folder='/project/src/dashboard/assets/')
app.title = "Aerial Ridesharing Routing"

#---- Dash app ----
heli_selection = [{"label":str(val), "value":val} for val in [k for k in range(2, 9)]]
skyport_selection = [{"label":str(val), "value":val} for val in [k for k in range(2, 9)]]
demand_selection = [{"label":str(val), "value":val} for val in [k * 10 for k in range(1, 9)]]

dict_heli = {k : [f"h{i+1}" for i in range(k)] for k in range(2, 9)}
def_cat = 2
def_index = 0


app.layout = html.Div(children=[
                      html.Div(className='row',
                               children=[
                                  html.Div(className='four columns div-user-controls',
                                  children=[
                                              html.Div(id='intermediate-value', style={'display': 'none'}),  #This div is hidden to stash simulated data
                                              html.Div(id='stashed-mh-sol', style={'display': 'none'}),  #This div is hidden to stash simulated data
                                              html.Div(id='stashed-mh-sol-bis', style={'display': 'none'}),  #This div is hidden to stash simulated data
                                              html.Div(id='stash-for-update', style={'display': 'none'}),  #This div is hidden to stash simulated data
                                              html.Div(id='up-data', style={'display': 'none'}),  #This div is hidden to stash simulated data
                                              html.H1('Aerial Ridesharing Routing'),
                                              #html.P('''Number of skyports :'''),
                                              html.Div(
                                                        className='div-for-dropdown',
                                                        children=[
                                                          html.H6('''Number of skyports :''', style={"display": "inline-block", "width":"50%"}),
                                                            dcc.Dropdown(id='skyport-selection',
                                                                          options= skyport_selection,
                                                                          value=2,
                                                                          style={'backgroundColor': '#1E1E1E', "display": "inline-block", "width":"50%", "vertical-align": "top"},
                                                                          ),
                                                        ],
                                                        style={'color': '#1E1E1E'}),
                                              #html.Div([ html.H6('''Number of Helicopters :''')], style={'float': 'right','margin': 'auto'}),
                                              html.Div(
                                                        className='div-for-dropdown',
                                                        children=[
                                                          html.H6('''Number of Helicopters :''', style={"display": "inline-block", "width":"50%"}),
                                                            dcc.Dropdown(id='helicopter-selection',
                                                                          options= heli_selection,
                                                                          value=2,
                                                                          style={'backgroundColor': '#1E1E1E', "display": "inline-block", "width":"50%", "vertical-align": "top"},
                                                                          ),
                                                        ],
                                                        style={'color': '#1E1E1E'}),

                                              #html.P('''Number of demands :'''),
                                              html.Div(
                                                        className='div-for-dropdown',
                                                        children=[
                                                          html.H6('''Number of demands :''', style={"display": "inline-block", "width":"50%"}),
                                                            dcc.Dropdown(id='demand-selection',
                                                                          options= demand_selection,
                                                                          value=10,
                                                                          style={'backgroundColor': '#1E1E1E', "display": "inline-block", "width":"50%", "vertical-align": "top"},
                                                                          ),
                                                        ],
                                                        style={'color': '#1E1E1E'}),
                                              html.Button('Simulate Data', id='button-sim', n_clicks=None),
                                              html.Button('Run MH', id='button', n_clicks=None),
                                              html.Button('Commit Change', id='button-stash', n_clicks=None),
                                              html.Button('Run MH Update', id='button-update', n_clicks=None),
                                              html.Div(
                                                        className='div-for-dropdown',
                                                        children=[
                                                          html.H6('''Simulate failure:''', style={"display": "inline-block", "width":"25%"}),
                                                          dcc.Dropdown(id='failure-heli',
                                                                          options=[{'label':l, 'value':l} for l in dict_heli[def_cat]],
                                                                          value=dict_heli[def_cat][def_index],
                                                                          style={'backgroundColor': '#1E1E1E', "display": "inline-block", "width":"30%", "vertical-align": "top"},
                                                                          ),
                                                          dcc.Input(id='failure_time', placeholder="HH:MM", type='text', style={"display": "inline-block", "width": "25%", 'backgroundColor': '#1E1E1E', "color": "white", "vertical-align": "top"}),
                                                          html.Button('Run', id='button-failure', n_clicks=None, style={"display": "inline-block", "vertical-align": "top"}),
                                                        ],
                                                        style={'color': '#1E1E1E'}),
                                              dcc.Tab(label="Metrics", className='pretty-tab', selected_className='pretty-tab--selected',
                                                        children=[
                                                                #html.H1('Metrics', style={"margin-top": "20px"}),
                                                                 html.Div(dcc.Markdown(id="time-simu"),
                                                                          style={"height": "100%", "margin-top": "20px"}),
                                                                html.Div(dcc.Markdown(id="Metrics"),
                                                                          style={"height": "100%", "margin-top": "10px"}),

                                                                html.Div(dcc.Graph(id="anytime"),
                                                                          style={"margin-top": "10px", "overflowY":'scroll', 'height':200})]),

                                              html.Div(children=[
                                                              dcc.Markdown(id='relayout-data'),

                                                          ]
                                                      )

                                                                ]),


                                  html.Div(className='eight columns div-for-charts bg-grey',
                                          #style = {"background-color":"#989898"},
                                          children=[html.Div(className="div-for-charts",
                                                            style = {"height":"100%"},
                                                            children=[
                                                              dcc.Tabs(parent_className='custom-tabs',
                                                              className='custom-tabs-container',
                                                              children =
                                                              [
                                                              dcc.Tab(label="Routing Graph", className='pretty-tab', selected_className='pretty-tab--selected', children=[
                                                              dcc.Graph(id='Graph', config={
                                                                                          'editable': True,
                                                                                          'edits': {
                                                                                              'shapePosition': True,
                                                                                              "axisTitleText": False,
                                                                                              "titleText":False
                                                                                          }
                                                                                      }),

                                                              ]),

                                                              dcc.Tab(label="Demands", className='pretty-tab', selected_className='pretty-tab--selected', children=[
                                                              dcc.Graph(id='Graph-init-demands', config={
                                                                                          'editable': False,

                                                                                      }),
                                                              ]),
                                                              dcc.Tab(label="Helicopters", className='pretty-tab', selected_className='pretty-tab--selected', children=[
                                                              dcc.Graph(id='heli-carac', config={
                                                                                          'editable': False,

                                                                                      }),
                                                              dcc.Markdown(id='imb-heli')
                                                              ])

                                                            ])
                                                            ]),

                                                    html.Div(className="div-for-charts", style={"position":"relative"}, children=[
                                                    dcc.Tabs(parent_className='custom-tabs',
                                                              className='custom-tabs-container',
                                                              children =
                                                              [
                                                              dcc.Tab(label="Routing Details", className='pretty-tab', selected_className='pretty-tab--selected', children=[
                                                                                                      html.H1('Routing Details'),
                                                                                                      html.Div(dcc.Markdown(id="Logs", style={"maxHeight": "400px", "overflow-y":"scroll"})),
                                                                                                      ]),
                                                              dcc.Tab(label="Skyports Configuration", className='pretty-tab', selected_className='pretty-tab--selected', children=[
                                                                                                      dcc.Graph(id='Graph-skyports')
                                                                                                      ]),
                                                              dcc.Tab(label="Metaheuristic Logs", className='pretty-tab', selected_className='pretty-tab--selected', children=[
                                                                                                      html.H1('Metaheuristic Full Logs'),
                                                                                                      html.Div(dcc.Markdown(id="MH-algo-logs", style={"maxHeight": "400px", "overflow-y":"scroll"})),
                                                                                                      ])

                                                    ])
                                                    ])



                                          ])
                                  ])
                                ])

ins_callbacks(app)

if __name__ == "__main__":
  app.run_server(debug=True, host='0.0.0.0')
