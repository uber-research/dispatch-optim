import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
#from dash.dependencies import Input, Output, State, Event
import sys
sys.path.append('/project/src/')
import numpy as np
import itertools
import operator
import matplotlib.pyplot as plt
import networkx as nx
import time
import utils as U
import matplotlib.image as mpimg
import copy
import heapq
import optim
from dash.exceptions import PreventUpdate
import json
import re
import dash_table
#import dash_bootstrap_components as dbc
import collections
from PIL import Image
import dash
import utils_expe as ue
img = Image.open('/project/src/dashboard/assets/icon.png')
img_heli = Image.open('/project/src/dashboard/assets/helicon_small.png')
np.random.seed(7)






def path_description(obj, paths_sol, A, A_s, A_g):
  #now read the path
  text =""
  for h in obj.helicopters:
      path = paths_sol[h]
      fuel_level = obj.carac_heli[h]["init_fuel"]
      text += f"  \n{h} starts with {round(100 * fuel_level / obj.carac_heli[h]['fuel_cap'], 2)}  % of fuel." #print(h, " starts with ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), " % of fuel.")
      served = 0
      for i in range(1, len(path)):
          prev, succ = path[i-1], path[i]
          a = (prev, succ)

          if not(a in A):
              continue
          if prev[0] != succ[0]:
              #change in location

              fuel_level -= obj.fly_time[(prev[0], succ[0])] * obj.carac_heli[h]["conso_per_minute"]
              if a in A_s:
                  served += 1
                  text += f"  \n{h} starts service in {prev[0]} at {'{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60))} and finishes in {succ[0]} at {'{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60))} - Fuel level at arrival: {round(100 * fuel_level / obj.carac_heli[h]['fuel_cap'], 2)} %" #print(h, " starts service in ", prev[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60)), "and finishes in ", succ[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60)), " - Fuel level at arrival: ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), "%", " - Minimum level reached " * (fuel_level < 325) )
                  if fuel_level < 325:
                    text += " - **Minimum level reached**"

              else:
                  #print(h, " leaves from ", prev[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60)), "and arrives in ", succ[0], "at ", '{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60)), " - Fuel level at arrival: ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), "%", " - Minimum level reached " * (fuel_level < 325))
                  text += f"  \n{h} leaves from  {prev[0]} at {'{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60))} and arrives in  {succ[0]} at {'{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60))} - Fuel level at arrival: {round(100 * fuel_level / obj.carac_heli[h]['fuel_cap'], 2)} %"
                  if fuel_level < 325:
                    text += " - **Minimum level reached**"

          else:
              if a in A_g:
                  #print(h, i, prev, succ)
                  fuel_level = obj.carac_heli[h]["fuel_cap"]
                  #print(h, " starts refueling in ", prev[0], " at ",'{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60)), ", finishes at ", '{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60)), " - Fuel level at arrival: ", round(100 * fuel_level / self.carac_heli[h]["fuel_cap"], 2), "%", " - Minimum level reached " * (fuel_level < 325))
                  text += f"  \n{h} starts refueling in  {prev[0]} at {'{:02d}:{:02d}'.format(*divmod(420 + prev[1], 60))} , finishes at  {'{:02d}:{:02d}'.format(*divmod(420 + succ[1], 60))} - Fuel level at arrival: {round(100 * fuel_level / obj.carac_heli[h]['fuel_cap'], 2)} %"
      text += "  \n"

  return text
      #print(served, " requests are served out of a total of ", len(self.r))




def get_graph(A, nodes, locations):
        """ Viz functions for the time expanded network, schedule is represented on graph.
          --------------
          Params :
                arcs : dict variable, contains arcs present in each helicopter's path
                A : list, contains all arcs
                A_g : list, contains all refuelling arcs
                A_s : list, contains all service arcs
                colors : dict, colors to represent helicopters on graph
                notes : str

          --------------
          Returns :
                fig : matplotlib figure to be plotted/saved

        """

        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n, pos=(n[1], locations.index(n[0])))
        if A is not None:
          G.add_edges_from(A)
        pos = nx.get_node_attributes(G, 'pos')
        return G, pos


def make_edge(x, y, text, width, color, annot=False):
    if annot:
      return  go.Scatter(x         = x,
                        y         = y,
                        line      = dict(width = width,
                                    color = color),
                        hoverinfo = "text",
                        text      = ([text if y[1] > x[1] + 1 else None]),
                        mode='lines+text',
                        showlegend=False)
    else:
      return  go.Scatter(x         = x,
                        y         = y,
                        line      = dict(width = width,
                                    color = color),
                        hoverinfo = "text",
                        text      = ([text if y[1] > x[1] + 1 else None]),
                        mode='lines',
                        showlegend=False)

def make_node(x, y, text, size):
    return go.Scatter(x = x,
                      y = y,
                      mode = 'markers',
                      marker_size = size,
                      text = text,
                     marker_color="black", showlegend=False)




def generate_plotly_fig(G, pos, helicopters, arcs, skyports, T, A_f, A_g, colors, r, carac_heli):
  """

  """
  edge_trace = []
  fuel = []
  for h in helicopters:
        if h in arcs:
          for e in arcs[h]:
            if not (e in A_f):
              x0, y0 = G.nodes[e[0]]['pos']
              x1, y1 = G.nodes[e[1]]['pos']

              trace = make_edge([x0, x1, None],
                                [y0, y1, None],
                                f"{h} serving",
                                4,
                                colors[h])
            if e in A_f:
              x0, y0 = G.nodes[e[0]]['pos']
              x1, y1 = G.nodes[e[1]]['pos']

              trace = make_edge([x0, x1, None],
                                [y0, y1, None],
                                f"{h} deadheading",
                                1,
                                colors[h])
            if e in A_g:
              fuel.append((e, colors[h]))

            edge_trace.append(trace)

  node_trace = go.Scatter(
      x=[],
      y=[],
      text=[],
      mode='markers',
      hoverinfo='text',
      showlegend=False,
      marker=dict(
          showscale=False,
          colorscale='YlGnBu',
          reversescale=True,
          color=[],
          size=2,
          colorbar=dict(
              thickness=0,
              title='',
              xanchor='center',
              titleside='top'
          ),
          line=dict(width=1)))

  for node in G.nodes():
      x, y = G.nodes[node]['pos']
      node_trace['x'] += tuple([x])
      node_trace['y'] += tuple([y])
  if len(skyports) < 5:
        height_graph = 180
  elif 5 <= len(skyports) < 9:
    height_graph = 100
  elif len(skyports) >= 9:
    height_graph = 900 / len(skyports)

  fig = go.Figure(
                  layout=go.Layout(
                      title={
                            'text': "",
                            'y':1,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'},
                      titlefont=dict(size=30),
                      showlegend=False,
                      height=height_graph*len(skyports),

                      paper_bgcolor="#eee5e5",
                      plot_bgcolor="#eee5e5",
                      hovermode='closest',

                      margin=dict(b=0,l=0,r=0,t=20),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=True, tickvals = [i for i in range(len(T))][::30], ticktext=['{:02d}:{:02d}'.format(*divmod(420 + i, 60)) for i in range(len(T))][::30]),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=True, tickvals = [i for i in range(len(skyports))], ticktext = skyports)))

  for trace in edge_trace:
    fig.add_trace(trace)

  fig.add_trace(node_trace)

  #--- Adding fuel icon to signal refueling
  offset = 0.03
  scale = 0.91
  for e in fuel:
    ed, c = e
    x1, y1 = G.nodes[ed[1]]['pos']
    x0, y0 = G.nodes[ed[0]]['pos']
    x = (x1 + x0)/2
    if y0 > 0:
      y = y0 - 0.1*y0
    else:
      y = y0
    print(f"Adding image in relative pos {((x / max(T)) + offset )*scale  , y/(len(skyports)-1)}")


    fig.add_layout_image(dict(
                              source=img,
                              opacity=1.0,
                              xref="paper", yref="paper",
                              x= ((x / max(T)) + offset )*scale , y= y/(len(skyports)-1),
                              sizex=0.09, sizey=0.09,
                              xanchor="left", yanchor="bottom",

                          )
                      )




  fig.update_layout(showlegend=True)

  fig.add_trace(go.Scatter(mode="markers", x=[None],
                                y=[None], marker_symbol="square",
                              marker_line_color="green", marker_color="green",
                              marker_line_width=2, marker_size=15, name = "Flights", showlegend=True,
                              hovertemplate=f"")
        )

  for req in r:

    #print(req, req[2], skyports.index(req[0]))
    fig.add_shape(
        # demand in plot
            type="rect",
            xref="paper",
            yref="paper",
            x0=req[2],
            y0=skyports.index(req[0]),
            x1=req[2]+5,
            y1=skyports.index(req[0]) + 0.02,
            name="Flights",

            line=dict(
                color="Green",
                width=8,
            ),
            fillcolor="Green",
        )
  fig.update_shapes(dict(xref='x', yref='y'))

  #--- Adding  icon for aircrafts
  offset = 0.03
  used = []
  for h in helicopters:
    start = carac_heli[h]["start"]
    y0 = skyports.index(start)
    if y0 > 0:
      y = y0 - 0.1 * y0
    else:
      y = y0
    x, y = (offset, y / (len(skyports) - 1) + offset)
    while [x, y] in used:
      y += 0.02


    fig.add_layout_image(dict(
                          source=img_heli,
                          opacity=1.0,
                          xref="paper", yref="paper",
                          x= x, y= y,
                          sizex=0.05, sizey=0.05,
                          xanchor="left", yanchor="bottom",

                      )
                  )
    # fig.add_shape(
    #     type="rect",
    #     x0=x - 20,
    #     y0=y+0.03,
    #     x1=x,
    #     y1=y-0.05,
    #     line=dict(
    #         color=colors[h],
    #         width=3,
    #     ),
    # )
    used.append([x, y])



  return fig








def generate_instance_params(l, d, h):
  colors = {"h1": "blue", "h2": "red", "h3": "orange", "h4":"yellow", "h5":"salmon", "h6": "pink", "h7":"olive", "h8":"darkred"}
  helicopters = [f"h{i}" for i in range(1, h+1)]
  skyports = [f"S{i}" for i in range(1, l + 1)]
  refuel_times = {loc: np.random.choice([int(25), int(15), int(5)], p=np.array([0.5, 0.3, 0.2])) for loc in skyports}
  prices = {int(25): int(100), int(15): int(200), int(5): int(700)}
  refuel_prices = {loc: prices[refuel_times[loc]] for loc in skyports}
  T = [int(k) for k in range(720)]  # timesteps

  fly_time = {}
  for s1, s2 in itertools.combinations(skyports, r = 2):
      fly_time[(s1, s2)] = np.random.choice([int(7), int(10), int(15)])
      fly_time[(s2, s1)] = fly_time[(s1, s2)]
  for s in skyports:
      fly_time[(s, s)] = int(0)
  fees = {loc: np.random.choice([int(200), int(300), int(100)]) for loc in skyports}
  parking_fee = {loc : np.random.choice([int(200), int(300), int(100), int(0)], p=np.array([0.5, 0.1, 0.3, 0.1])) for loc in skyports}
  beta = 500
  min_takeoff = 325
  n_iter = 10000
  pen_fuel = 0
  no_imp_limit = 1600
  nodes = list(itertools.product(skyports, T))

  carac_heli = {h : {
          "cost_per_min": int(34),
          "start": np.random.choice(skyports),
          "security_ratio": int(2),
          "conso_per_minute": int(20),
          "fuel_cap": int(1100),
          "init_fuel": np.random.choice([int(900), int(1100), int(1000)]),
          "theta": int(1000)}  for h in helicopters}


  n_request = d

  # sink nodes for helicopters
  sinks = {h: ("sink %s" % h, max(T)) for h in helicopters}

  # Simulate requests
  r = U.simulate_requests(n_request, fly_time, skyports, T, verbose=0)

  A_w, A_s, A_f, A_g = U.create_arcs(nodes, r, fly_time, refuel_times)

  A = A_w + A_s + A_f + A_g
  return A, A_f, A_s, A_g, nodes, carac_heli, skyports, helicopters, refuel_prices, refuel_times, colors, fly_time, fees, parking_fee, beta, min_takeoff, n_iter, pen_fuel, no_imp_limit, r, sinks, T



def create_anytime_table(perf_over_time, best):

    fig = go.Figure(data=[go.Table(
      columnwidth = [14, 8, 10],
      header=dict(values=['<b>Running Time (seconds)</b>', '<b>Cost ($)</b>', '<b>Gap to best (%)</b>'],
                  line_color='#eee5e5',
                  fill_color='#eee5e5',
                  align='left',
                  font=dict(size=15, color="black", family="Open Sans Light")),
      cells=dict(values=[[perf_over_time[i][1] for i in range(len(perf_over_time))], # 1st column
                        [round(perf_over_time[i][0], 2) for i in range(len(perf_over_time))],
                        [round(100 * (perf_over_time[i][0] - best) / best, 2) for i in range(len(perf_over_time))]],
                line_color='#eee5e5',
                #fill_color = col,
                fill_color='#eee5e5',
                font=dict(color='black', family="Open Sans Light", size=12),
                align='center'))
    ])
    fig.update_layout(paper_bgcolor="#eee5e5", title_x=0., margin = {'l': 0, 'r': 0, 't': 5, 'b': 3}, title_font_color="black", title_font_family="Open Sans Light")
    return fig

def design_initial_demands(skyports, T, fly_time, r, nodes, colors, helicopters, carac_heli):
      """
      """
      G, pos = get_graph(None, nodes, skyports)
      node_trace = go.Scatter(
          x=[],
          y=[],
          text=[],
          mode='markers',
          hoverinfo='text',
          showlegend=False,
          marker=dict(
              showscale=False,
              colorscale='YlGnBu',
              reversescale=True,
              color=[],
              size=2,
              colorbar=dict(
                  thickness=0,
                  title='',
                  xanchor='center',
                  titleside='top'
              ),
              line=dict(width=1)))

      for node in G.nodes():
          x, y = G.nodes[node]['pos']
          node_trace['x'] += tuple([x])
          node_trace['y'] += tuple([y])
      #------- plotly fig for the TS network -----
      if len(skyports) < 5:
        height_graph = 200
      elif 5 <= len(skyports) < 9:
        height_graph = 100
      elif len(skyports) >= 9:
        height_graph = 900 / len(skyports)

      fig = go.Figure(
                      layout=go.Layout(
                          title={
                                'text': "",
                                'y':1,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'},
                          titlefont=dict(size=30),
                          showlegend=False,
                          height=height_graph * len(skyports),
                          paper_bgcolor="#eee5e5",
                          plot_bgcolor="#eee5e5",
                          hovermode='closest',
                          margin=dict(b=50,l=20,r=5,t=50),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=True, tickvals = [i for i in range(len(T))][::30], ticktext=['{:02d}:{:02d}'.format(*divmod(420 + i, 60)) for i in range(len(T))][::30]),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=True, tickvals = [i for i in range(len(skyports))], ticktext = skyports)))

      fig.add_trace(node_trace)
      fig.update_layout(showlegend=True)
      #-- Adding demands slot
      idx = 0
      sl = True
      for req in r:
        idx += 1
        if idx > 1:
          sl=False
        pax = np.random.choice([4, 5, 6])
        #print(req, req[2], skyports.index(req[0]))
        fig.add_trace(go.Scatter(mode="markers", x=[req[2]],
                                y=[skyports.index(req[0])], marker_symbol="square",
                              marker_line_color="green", marker_color="green",
                              marker_line_width=2, marker_size=15, name = "Flights", showlegend=sl, legendgroup="F",
                              hovertemplate=f"ID: {idx}<br>PAX: {pax}<br>Origin: {req[0]}<br>Destination: {req[1]}<br>Departure Time: {'{:02d}:{:02d}'.format(*divmod(420 + req[2] + 5, 60))}")
        )
      offset = 0.03
      used = []
      for h in helicopters:
        start = carac_heli[h]["start"]
        y0 = skyports.index(start)
        if y0 > 0:
          y = y0 - 0.1 * y0
        else:
          y = y0
        x, y = (offset, y / (len(skyports) - 1) + offset)
        while [x, y] in used:
          y += 0.02


        fig.add_layout_image(dict(
                              source=img_heli,
                              opacity=1.0,
                              xref="paper", yref="paper",
                              x= x, y= y,
                              sizex=0.05, sizey=0.05,
                              xanchor="left", yanchor="bottom",

                          )
                      )

        used.append([x, y])


      return fig




def update_skyport_graph_config(skyports, refuel_times, refuel_prices, T, fly_time, fees, parking_fee):
      """
      """

      print("Updating skyports config...")
      #Creates Graph before plotting
      G = nx.Graph()
      for s in skyports:
          G.add_node(s)
      for s1 in skyports:
          for s2 in skyports:
            if s1 != s2 and (s1, s2) not in G.edges():
              G.add_edge(s1, s2)
      #assign positions
      pos = nx.random_layout(G)
      edge_trace = []
      annot = []
      done = []
      for e in G.edges:
          x0, x1 = pos[e[0]][0], pos[e[1]][0]
          y0, y1 = pos[e[0]][1], pos[e[1]][1]
          trace = make_edge([x0, x1, None],
                          [y0, y1, None],
                          f"Travel time : {fly_time[(e[0], e[1])]} from {e[0]} to {e[1]}",
                          3,
                          "black",
                          annot=True)
          edge_trace.append(trace)

          if (e[0], e[1]) not in done:
              annot.append([(x0 + x1)/2, (y0 + y1)/2, f"Fly time : {fly_time[(e[0], e[1])]} minutes."])
              done.append((e[0], e[1]))



      node_trace = []

      for node in G.nodes():
          x, y = pos[node][0], pos[node][1]
          desc = f"Location : {node}<br>Refuel price : {refuel_prices[node]}<br>Refuel time : {refuel_times[node]}<br>Parking fee : {parking_fee[node]}<br>Landing fee : {fees[node]}"
          trace = make_node([x], [y], desc, 30)
          node_trace.append(trace)



      fig = go.Figure(
                    layout=go.Layout(
                        title={
                              'text': "",
                              'y':1,
                              'x':0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'},
                        titlefont=dict(size=30),
                        showlegend=False,
                        height=100 * len(skyports),
                        paper_bgcolor="#eee5e5",
                        plot_bgcolor="#eee5e5",
                        hovermode='closest',

                        margin=dict(b=50,l=20,r=5,t=50),
                      xaxis=dict(showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(showgrid=False, zeroline=False, visible=False),
                    ))


      for trace in edge_trace:
          fig.add_trace(trace)
      for trace in node_trace:
          fig.add_trace(trace)

      for an in annot:
          fig.add_annotation(
            x=an[0],
            y=an[1],
            text=an[2])
          #ho = make_node([an[0]], [an[1]], an[2], 2)
          #fig.add_trace(ho)
      fig.update_annotations(dict(
            xref="x",
            yref="y",
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40
        ))

      fig.update_layout(showlegend=False)

      print("Skyport config updated")
      return fig




def key_to_json(data):
    if data is None or isinstance(data, (bool, int, str)):
        return data
    if isinstance(data, (tuple, frozenset)):
        return str(data)
    if isinstance(data, (np.integer, np.int64)):
            return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    raise TypeError(f"Object if of type {type(data)}")

def to_json(data):
    if data is None or isinstance(data, (bool, int, range, str, list)):
        return data
    if isinstance(data, (set, frozenset)):
        return sorted(data)
    if isinstance(data, np.int64):
        return int(data)
    if isinstance(data, tuple):
        return str(data)
    if isinstance(data, dict):
        return {key_to_json(key): to_json(data[key]) for key in data}
    raise TypeError(f"Object is of type {type(data)}")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key_to_json(key): to_json(obj) for key in obj}
        else:
            return super(NpEncoder, self).default(obj)


class Decoder(json.JSONDecoder):
    def decode(self, s):
        result = super().decode(s)  # result = super(Decoder, self).decode(s) for Python 2.x
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, str):
            try:
                return int(o)
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {k: self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self._decode(v) for v in o]
        else:
            return o


import ast
def process_stashed_dict(sd):
  res = {}
  for k in sd.keys():
    u = ast.literal_eval(k)
    res[u] = sd[k]
  return res

def list_to_tuple(ls):

  res = []
  for ele in ls:
    if type(ele[0]) == list:
      res.append(tuple(tuple(x) for x in ele))
    else:
      res.append(tuple(ele))
  return res

def process_req_special(rjs):
    res = {}
    for k in rjs.keys():
        u = ast.literal_eval(k)
        li = list(tuple(tuple(list_to_tuple(x)) for x in rjs[k]))
        res[u] = li
    return res




def ins_callbacks(app):
  """

  """

  @app.callback([
                Output('Graph-skyports', 'figure'),
                Output('Graph-init-demands', 'figure'),
                Output('intermediate-value', 'children'),
                Output('time-simu', 'children')
                ],
                [Input('button-sim', 'n_clicks'),
                  Input('skyport-selection', 'value'),
                  Input('demand-selection', 'value'),
                  Input('helicopter-selection', 'value'),
                ])
  def simulate_data(n_clicks, l, d, h):
    """
    """
    if n_clicks is None:
        raise PreventUpdate
    else:
      start = time.time()
      A, A_f, A_s, A_g, nodes, carac_heli, skyports, helicopters, refuel_prices, refuel_times, colors, fly_time, fees, parking_fee, beta, min_takeoff, n_iter, pen_fuel, no_imp_limit, r, sinks, T = generate_instance_params(l, d, h)
      #--- Stashing Data in hidden div ---

      sim_data = {
                  "A": A,
                  "A_f": A_f,
                  "A_s": A_s,
                  "A_g": A_g,
                  "nodes": nodes,
                  "carac_heli": carac_heli,
                  "skyports": skyports,
                  "helicopters": helicopters,
                  "refuel_prices": refuel_prices,
                  "refuel_times": refuel_times,
                  "colors": colors,
                  "fly_time": to_json(fly_time),
                  "fees": fees,
                  "parking_fee": parking_fee,
                  "beta": int(beta),
                  "min_takeoff": int(min_takeoff),
                  "n_iter": int(n_iter),
                  "pen_fuel": int(pen_fuel),
                  "no_imp_limit": int(no_imp_limit),
                  "r":to_json(r),
                  "sinks": sinks,
                  "T": T
      }

    fig_skyports = update_skyport_graph_config(skyports, refuel_times, refuel_prices, T, fly_time, fees, parking_fee)
    fig_init_demands = design_initial_demands(skyports, T, fly_time, r, nodes, colors, helicopters, carac_heli)
    time_sim = f"Time to simulate data and create initial plots : {round(time.time() - start, 2)} seconds."
    return fig_skyports, fig_init_demands, json.dumps(sim_data, cls=NpEncoder), time_sim




  @app.callback([Output("stashed-mh-sol-bis", "children")
                ],
                [Input('button-stash', 'n_clicks')]
                ,
                [State('stashed-mh-sol', 'children')
                ])
  def stash_sol_manager(n_clicks, stashed_sol):
    """
    """
    if n_clicks is None:
        raise PreventUpdate
    else:
      print("Stashing sol annex")
      l = ast.literal_eval(stashed_sol)
      return [json.dumps(l)]



  @app.callback([Output('Metrics', 'children'),
                Output('Logs', 'children'),
                Output('Graph', 'figure'),
                Output("stashed-mh-sol", "children"),
                Output("MH-algo-logs", "children"),
                Output("anytime", "figure"),
                Output('up-data', 'children')
                ],
                [Input('button', 'n_clicks'),
                  Input('button-update', 'n_clicks'),
                  Input('button-failure', 'n_clicks'),
                ],
                [State('intermediate-value', 'children'),
                  State('stashed-mh-sol-bis', 'children'),
                  State('stash-for-update', 'children'),
                  State('up-data', 'children'),
                  State('failure_time', 'value'),
                  State('failure-heli', 'value'),
                  State('stashed-mh-sol', 'children')])
  def run_mh(button_run, button_update, button_failure, stashed_data, stashed_sol, data_info, new_data, fail_time, failure_heli, init_sol):
    """
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == "button-sim":
          return None, None, None, None, None, None

        elif button_id == "button":
          data = json.loads(stashed_data, cls=Decoder)
          #A, A_f, A_s, A_g, nodes, carac_heli, skyports, helicopters, refuel_prices, refuel_times, colors, fly_time, fees, parking_fee, beta, min_takeoff, n_iter, pen_fuel, no_imp_limit, r, sinks, T = generate_instance_params(l, d, h)
          #Actually run the MH
          start = time.time()
          requests = process_req_special(data["r"])
          print("Starting to solve...")
          A = list_to_tuple(data["A"])
          A_f = list_to_tuple(data["A_f"])
          A_s = list_to_tuple(data["A_s"])
          A_g = list_to_tuple(data["A_g"])
          nodes = list_to_tuple(data["nodes"])
          #print(A_s)


          meta = optim.META_HEURISTIC(
              requests,
              data["helicopters"],
              data["carac_heli"],
              data["refuel_times"],
              data["refuel_prices"],
              data["skyports"],
              nodes,
              data["parking_fee"],
              data["fees"],
              process_stashed_dict(data["fly_time"]),
              data["T"],
              data["beta"],
              data["min_takeoff"],
              data["pen_fuel"],
              A_s,
              A,
              A_g)
          meta.init_encoding()
          meta.init_compatibility()
          meta.init_request_cost()

          init_heuri = meta.init_heuristic_random()
          best_sol, best_cost, _, _, perf_over_time, dist, log_mh = meta.VNS(
              init_heuri, data["n_iter"], eps=0.12, no_imp_limit=data["no_imp_limit"], random_restart=200, verbose=0)
          print("Metaheuristic finished. Now starting to generate figures...")
          dur = round(time.time() - start, 2)
          valid = meta.fuel_checker(best_sol)
          unserved, served = meta.update_served_status(best_sol)
          text_metrics = f"  \n**Metrics**   \nNumber of served demands : {len(served)} out of {len(meta.r)}.  \nMetaheuristic finished in {dur} seconds.  \nSolution cost is ${best_cost}"
          _, cost_pen, _ = meta.compute_cost(best_sol)
          paths_sol = meta.descibe_sol(best_sol, A_s, A_g, A, notes="", verbose=0)
          text_log = path_description(meta, paths_sol, A, A_s, A_g)
          arcs = meta.get_arcs(paths_sol)
          #--- add metrics
          add_metrics = meta.cost_breakdown(best_sol, paths_sol, A, A_s, A_g)
          text_metrics += f" including {add_metrics['nb_dh']} deadheads (${add_metrics['cost_dh']}), {add_metrics['nb_ref']} refuels (${add_metrics['cost_ref']})  \nand {len(unserved)} unserved demands (${cost_pen}).  \nLower bound on the number of refuel needed here : {add_metrics['min_nb_ref']} "

          G, pos = get_graph(A, nodes, data["skyports"])
          fig = generate_plotly_fig(G, pos, data["helicopters"], arcs, data["skyports"], data["T"], A_f, A_g, data["colors"], requests, data["carac_heli"])
          print("Done.")
          print("")
          print("Stashing solution.")
          fig_anytime = create_anytime_table(perf_over_time, best_cost)
          # text_anytime = ""
          # for ele in perf_over_time:
          #     text_anytime += f"  \nMetaheuristic found solution with cost {ele[0]} $ in {ele[1]} seconds."
          return text_metrics, text_log, fig, json.dumps(best_sol.tolist()), log_mh, fig_anytime, json.dumps(data, cls=NpEncoder)# text_anytime

        elif button_id == "button-update":
          print("Updating MH button triggered. Trying to retrieve stashed solution.")
          info = json.loads(data_info)
          action = info["action"]
          old = action[1]
          new_req = info["new_req"]
          sol = json.loads(stashed_sol)

          #sol = ast.literal_eval(sol)
          new_requests = process_req_special(info["requests"])

          #data = json.loads(stashed_data, cls=Decoder)
          init_data = json.loads(stashed_data, cls=Decoder)
          data = json.loads(new_data, cls=Decoder)
          #A, A_f, A_s, A_g, nodes, carac_heli, skyports, helicopters, refuel_prices, refuel_times, colors, fly_time, fees, parking_fee, beta, min_takeoff, n_iter, pen_fuel, no_imp_limit, r, sinks, T = generate_instance_params(l, d, h)
          #Actually run the MH
          requests = process_req_special(data["r"])
          A = list_to_tuple(data["A"])
          A_f = list_to_tuple(data["A_f"])
          A_s = list_to_tuple(data["A_s"])
          A_g = list_to_tuple(data["A_g"])
          nodes = list_to_tuple(data["nodes"])

          print("Running MH update...")
          meta = optim.META_HEURISTIC(
              requests,
              data["helicopters"],
              data["carac_heli"],
              data["refuel_times"],
              data["refuel_prices"],
              data["skyports"],
              nodes,
              data["parking_fee"],
              data["fees"],
              process_stashed_dict(data["fly_time"]),
              data["T"],
              data["beta"],
              data["min_takeoff"],
              data["pen_fuel"],
              A_s,
              A,
              A_g)
          meta.init_encoding()
          meta.init_compatibility()
          meta.init_request_cost()

          #--- compute assignement demands to heli to update encoding after user edit
          lon = 2 * len(requests)
          served_by = {h: [] for h in data["helicopters"]}
          ref = {h: [] for h in data["helicopters"]}
          rf0 = {h: False for h in data["helicopters"]}
          for i in range(len(data["helicopters"])):
            arr = sol[lon * i:lon * (i + 1)]
            if arr[0] == 1:
              rf0[data["helicopters"][i]] = True
            for j in range(len(arr)):
              if j % 2 != 0 and arr[j] == 1:
                #print(data["helicopters"][i], list(requests.keys())[j])
                served_by[data["helicopters"][i]].append(meta.indices[j])
                if j + 1 < len(arr) and arr[j + 1] == 1:
                  ref[data["helicopters"][i]].append(meta.indices[j])

          # print(served_by)
          if not (action[0]):
            # print("Updating without removal - No re-opt.")
            # print("Updating costs and logs...")
            #print(f"Old is {tuple(old)} and new {tuple(new_req)}")
            #print(f"Refuel after {ref}")
            start = time.time()
            meta.r = new_requests
            #print(meta.r)
            meta.init_compatibility()
            meta.init_request_cost()
            meta.init_encoding()
            A_w, A_s, A_f, A_g = U.create_arcs(meta.nodes, meta.r, meta.fly_time, meta.refuel_time)

            A = A_w + A_s + A_f + A_g
            #print("--")
            #print(meta.refuel_compatible)
            #print("--")
            #create new solutions with new encoding using the old one
            new_sol = [0] * len(sol)

            for h in data["helicopters"]:
              if rf0[h]:
                new_sol[lon * data["helicopters"].index(h)] = 1
              for rs in served_by[h]:
                rs = tuple(rs)
                if rs == tuple(old):
                  rs = tuple(new_req)
                new_sol[meta.reverse_indices[h][rs]] = 1
                if rs in ref[h]:
                  new_sol[meta.reverse_indices[h][rs] + 1] = 1
                if tuple(old) in ref[h] and rs == tuple(new_req):
                  new_sol[meta.reverse_indices[h][rs] + 1] = 1
            new_sol = np.array(new_sol)
            #---
            unserved, served = meta.update_served_status(new_sol)
            cost_heli, cost_pen, _ = meta.compute_cost(new_sol)

            #print(new_sol[meta.assign["h1"]])
            #print(cost_heli)
            best_cost = cost_pen
            for h in cost_heli:
                best_cost += cost_heli[h][0] + 2000 * cost_heli[h][1]
            #---- need to transalate previous sol into new one
            text_metrics = f"  \n**Metrics**  \nNumber of served demands : {len(served)} out of {len(meta.r)}.  \nMetaheuristic finished updating in {round(time.time() - start, 2)} seconds.  \nSolution cost is ${best_cost}."

            paths_sol = meta.descibe_sol(new_sol, A_s, A_g, A, notes="", verbose=0)
            #--- add metrics
            add_metrics = meta.cost_breakdown(new_sol, paths_sol, A, A_s, A_g)
            text_metrics += f" including {add_metrics['nb_dh']} deadheads (${add_metrics['cost_dh']}), {add_metrics['nb_ref']} refuels (${add_metrics['cost_ref']})  \nand {len(unserved)} unserved demands (${cost_pen}).  \nLower bound on the number of refuel needed here : {add_metrics['min_nb_ref']} "

            text_log = path_description(meta, paths_sol, A, A_s, A_g)
            arcs = meta.get_arcs(paths_sol)
            G, pos = get_graph(A, nodes, data["skyports"])
            fig = generate_plotly_fig(G, pos, data["helicopters"], arcs, data["skyports"], data["T"], A_f, A_g, data["colors"], meta.r, data["carac_heli"])
            #---- updating data
            up_data = {
                  "A": A,
                  "A_f": A_f,
                  "A_s": A_s,
                  "A_g": A_g,
                  "nodes": meta.nodes,
                  "carac_heli": meta.carac_heli,
                  "skyports": meta.locations,
                  "helicopters": meta.helicopters,
                  "refuel_prices": meta.refuel_price,
                  "refuel_times": meta.refuel_time,
                  "colors": init_data["colors"],
                  "fly_time": to_json(meta.fly_time),
                  "fees": meta.landing_fee,
                  "parking_fee": meta.parking_fee,
                  "beta": int(meta.beta),
                  "min_takeoff": int(meta.mintakeoff),
                  "n_iter": init_data["n_iter"],
                  "pen_fuel": int(0),
                  "no_imp_limit": init_data["no_imp_limit"],
                  "r":to_json(meta.r),
                  "sinks": init_data["sinks"],
                  "T": meta.T
                  }
            fig_anytime = create_anytime_table([(best_cost, 0)], best_cost)

            return text_metrics, text_log, fig, json.dumps(sol), "", fig_anytime, json.dumps(up_data, cls=NpEncoder)

          else:
            #remove requests
            print("Updating with removal - Reoptimization happenning")
            start = time.time()
            sol, _ = meta.remove_request(np.array(sol), tuple(action[1]))
            meta.r = new_requests
            A_w, A_s, A_f, A_g = U.create_arcs(meta.nodes, meta.r, meta.fly_time, meta.refuel_time)

            A = A_w + A_s + A_f + A_g
            meta.init_encoding()
            meta.init_compatibility()
            meta.init_request_cost()
            meta.init_encoding()

            best_sol, best_cost, _, _, perf_over_time, dist, log_mh = meta.VNS(
              np.array(sol, dtype='b'), data["n_iter"], eps=0.12, no_imp_limit=data["no_imp_limit"], random_restart=200, verbose=0)
            print("Metaheuristic finished update. Now starting to generate figures...")
            dur = round(time.time() - start, 2)
            unserved, served = meta.update_served_status(best_sol)
            text_metrics = f"  \n**Metrics**  \nNumber of served demands : {len(served)} out of {len(meta.r)}.  \nMetaheuristic finished updating in {dur} seconds.  \nSolution cost is ${best_cost}."
            paths_sol = meta.descibe_sol(best_sol, A_s, A_g, A, notes="", verbose=0)
            #--- add metrics
            _, cost_pen, _ = meta.compute_cost(best_sol)

            add_metrics = meta.cost_breakdown(best_sol, paths_sol, A, A_s, A_g)
            text_metrics += f" including {add_metrics['nb_dh']} deadheads (${add_metrics['cost_dh']}), {add_metrics['nb_ref']} refuels (${add_metrics['cost_ref']})  \nand {len(unserved)} unserved demands (${cost_pen}).  \nLower bound on the number of refuel needed here : {add_metrics['min_nb_ref']} "

            #---
            text_log = path_description(meta, paths_sol, A, A_s, A_g)
            arcs = meta.get_arcs(paths_sol)
            G, pos = get_graph(A, nodes, data["skyports"])
            fig = generate_plotly_fig(G, pos, data["helicopters"], arcs, data["skyports"], data["T"], A_f, A_g, data["colors"], meta.r, data["carac_heli"])

            fig_anytime = create_anytime_table(perf_over_time, best_cost)

            #---- updating data
            up_data = {
                  "A": A,
                  "A_f": A_f,
                  "A_s": A_s,
                  "A_g": A_g,
                  "nodes": meta.nodes,
                  "carac_heli": meta.carac_heli,
                  "skyports": meta.locations,
                  "helicopters": meta.helicopters,
                  "refuel_prices": meta.refuel_price,
                  "refuel_times": meta.refuel_time,
                  "colors": init_data["colors"],
                  "fly_time": to_json(meta.fly_time),
                  "fees": meta.landing_fee,
                  "parking_fee": meta.parking_fee,
                  "beta": int(meta.beta),
                  "min_takeoff": int(meta.mintakeoff),
                  "n_iter": init_data["n_iter"],
                  "pen_fuel": int(0),
                  "no_imp_limit": init_data["no_imp_limit"],
                  "r":to_json(meta.r),
                  "sinks": init_data["sinks"],
                  "T": meta.T
                  }


            return text_metrics, text_log, fig, json.dumps(best_sol.tolist()), log_mh, fig_anytime, json.dumps(up_data, cls=NpEncoder)

        elif button_id == "button-failure":
          stime = fail_time.split(":")
          hours = stime[0]
          minutes = stime[1]
          failure_time = (int(hours) - 7) * 60 + int(minutes)
          print(f"Starting to simulate failure.. for helicopter {failure_heli} at time {failure_time}.")

          data = json.loads(new_data, cls=Decoder)
          sol = json.loads(init_sol)
          sol = np.array(sol)
          #Actually run the MH
          start = time.time()
          requests = process_req_special(data["r"])

          A = list_to_tuple(data["A"])
          A_f = list_to_tuple(data["A_f"])
          A_s = list_to_tuple(data["A_s"])
          A_g = list_to_tuple(data["A_g"])
          nodes = list_to_tuple(data["nodes"])
          meta = optim.META_HEURISTIC(
              requests,
              data["helicopters"],
              data["carac_heli"],
              data["refuel_times"],
              data["refuel_prices"],
              data["skyports"],
              nodes,
              data["parking_fee"],
              data["fees"],
              process_stashed_dict(data["fly_time"]),
              data["T"],
              data["beta"],
              data["min_takeoff"],
              data["pen_fuel"],
              A_s,
              A,
              A_g)
          meta.init_encoding()
          meta.init_compatibility()
          meta.init_request_cost()

          # --- simulating failure
          new_sol, frozen_bits, _ = ue.sim_failure(meta, failure_heli, failure_time, sol)
          # --- starting to solve..
          print(new_sol)
          print(type(new_sol))
          print(new_sol[meta.assign["h2"]])
          print(f"New sol is feasible : {meta.feasible(new_sol)}")
          best_sol, best_cost, _, _, perf_over_time, _, log_mh = meta.VNS(
              np.array(new_sol, dtype='b'), data["n_iter"], eps=0.12, no_imp_limit=data["no_imp_limit"], random_restart=200, verbose=0, frozen_bits=frozen_bits)
          print("Metaheuristic finished. Now starting to generate figures...")
          dur = round(time.time() - start, 2)
          valid = meta.fuel_checker(best_sol)
          unserved, served = meta.update_served_status(best_sol)
          text_metrics = f"  \n**Metrics**   \nNumber of served demands : {len(served)} out of {len(meta.r)}.  \nMetaheuristic finished in {dur} seconds.  \nSolution cost is ${best_cost}"
          _, cost_pen, _ = meta.compute_cost(best_sol)
          paths_sol = meta.descibe_sol(best_sol, A_s, A_g, A, notes="", verbose=0)
          text_log = path_description(meta, paths_sol, A, A_s, A_g)
          arcs = meta.get_arcs(paths_sol)
          #--- add metrics
          add_metrics = meta.cost_breakdown(best_sol, paths_sol, A, A_s, A_g)
          text_metrics += f" including {add_metrics['nb_dh']} deadheads (${add_metrics['cost_dh']}), {add_metrics['nb_ref']} refuels (${add_metrics['cost_ref']})  \nand {len(unserved)} unserved demands (${cost_pen}).  \nLower bound on the number of refuel needed here : {add_metrics['min_nb_ref']} "

          G, pos = get_graph(A, nodes, data["skyports"])
          fig = generate_plotly_fig(G, pos, data["helicopters"], arcs, data["skyports"], data["T"], A_f, A_g, data["colors"], requests, data["carac_heli"])
          print("Done.")
          print("")
          print("Stashing solution.")
          fig_anytime = create_anytime_table(perf_over_time, best_cost)
          # text_anytime = ""
          # for ele in perf_over_time:
          #     text_anytime += f"  \nMetaheuristic found solution with cost {ele[0]} $ in {ele[1]} seconds."
          return text_metrics, text_log, fig, json.dumps(best_sol.tolist()), log_mh, fig_anytime, json.dumps(data, cls=NpEncoder)# text_anytime


  # def kill_update():
  #   pass


  @app.callback([
                  Output('relayout-data', 'children'),
                  Output("stash-for-update", "children")],
                [Input('Graph', 'relayoutData'),
                  Input('up-data', 'children'),
                  Input("stashed-mh-sol", "children")],
                [State('relayout-data', 'children'),
                 State("stash-for-update", "children")])
  def monitor_user_edits(relayoutData, data, sol, state_data, state_sol):
      print("Movement detected....")
      #print(relayoutData, type(relayoutData))
      if relayoutData is None:
        return state_data, state_sol
      if len(relayoutData) < 2:
        return state_data, state_sol
      print("Movement detected and action triggered...")

      #First retrieve original demands
      sol = ast.literal_eval(sol)
      data = json.loads(data, cls=Decoder)
      requests = process_req_special(data["r"])

      refuel_times = data["refuel_times"]
      fly_time = process_stashed_dict(data["fly_time"])
      text = "  \n**Editing Flights**:"
      # -- Getting the request that has been moved
      #demand_moved = int(list(relayoutData.keys())[0].split(".")[0][-2:-1])
      #demand_moved = int(re.search('[(.*)]', list(relayoutData.keys())[0].split(".")[0]).group(1))
      start = "["
      end = "]"
      s = list(relayoutData.keys())[0]
      demand_moved = int(s.split(start)[1].split(end)[0])
      print(demand_moved, relayoutData.keys())
      req_moved = list(requests.keys())[demand_moved]
      # -- Get the slack+ associated with this demand
      #helicopters = data["helicopters"]
      lon = 2 * len(requests)
      idr = 2 * demand_moved + 1
      print(idr)
      h = 0
      while sol[idr + h * lon] != 1:
        h += 1
        if h >= len(data["helicopters"]):
          print("Wrong trigger, aborting update")
          return state_data, state_sol
      ids = idr + h * lon
      rf = (sol[ids + 1] == 1)
      inr = None
      try:
          nxt = next(i for i in range(ids+2, len(sol)) if sol[i] == 1)
      except StopIteration:
          nxt = None
      # print(f"Next is {nxt}")
      if nxt is None:
          slack = max(0, 720 - requests[req_moved][-1][1][1])
      else:
          if nxt // lon == h:
            #find requests associated with nxt
            inr = int((nxt % lon - 1) / 2)
            req_nxt = list(requests.keys())[inr]
            slack = max(req_nxt[2] - requests[req_moved][-1][1][1]  - int(rf) * refuel_times[req_moved[1]] - fly_time[(req_moved[1], req_nxt[0])], 0)
          else:
            slack = max(720 - requests[req_moved][-1][1][1] - int(rf) * refuel_times[req_moved[1]], 0)


      print(f"Demand moved is {demand_moved}")
      print(f"Next found demand is {inr}")
      print(f"Slack time is {slack}")
      for k in relayoutData:
        #template key = "shapes[1].x0"
        if "x0" in k:
          movement_size = round(relayoutData[k] - req_moved[2])
          text += f"  \nDelaying demand {demand_moved} by {movement_size} minutes."
          if movement_size >= slack:
            text += "  \n**Moving past the slack time** - Current schedule will be impacted."

      new_requests = {}
      for req in requests:
        if req != req_moved:
          new_requests[req] = requests[req]
        else:
          new_requests[(req[0], req[1], req[2] + movement_size)] = [((ele[0][0] , ele[0][1] + movement_size ), (ele[1][0] , ele[1][1] + movement_size )) for ele in requests[req]]
          new_req = (req[0], req[1], req[2] + movement_size)


      rm = (False, req_moved)
      if "Moving past" in text:
        rm = (True, req_moved)
      #print(new_requests)
      actions = {"action":rm, "sol": sol, "requests":to_json(new_requests), "new_req":new_req}
      return text, json.dumps(actions, cls=NpEncoder)







  @app.callback([Output("heli-carac", "figure"),
                Output("imb-heli", "children")
                ],
                [Input('stashed-mh-sol', 'children')],
                [State('intermediate-value', 'children')]
                )
  def update_heli_status(sol, data):
    """
    """
    data = json.loads(data, cls=Decoder)
    requests = process_req_special(data["r"])
    heli = data["helicopters"]
    carac = data["carac_heli"]
    sol = ast.literal_eval(sol)
    #--- compute assignement demands to heli
    lon = 2 * len(requests)
    served_by = {h:[] for h in heli}
    for i in range(len(heli)):
      arr = sol[lon * i:lon * (i + 1)]
      for j in range(len(arr)):
        if j % 2 != 0 and arr[j] == 1:
          served_by[heli[i]].append(j)

    #---- compute imbalance metric
    nb_served = []
    for h in served_by:
      nb_served.append(len(served_by[h]))

    mu = np.mean(nb_served)
    std = np.std(nb_served)
    imb = f" **Imbalance measure** : {round(std / mu, 2)}"
    #--- Generate table
    col = []
    for k in range(8):
      col.append(['#eee5e5' for h in heli])
    col.append([data["colors"][h] for h in heli])
    fig = go.Figure(data=[go.Table(
      columnwidth = [75] * 8 + [25],
      header=dict(values=['<b>Helicopter ID</b>', '<b>Fuel Capacity</b>', '<b>Starting Fuel</b>',
                        "<b>Start</b>", '<b>Fuel Consumption</b>', '<b>Cost</b>',
                          '<b>Capacity</b>', '<b>Served Demands</b>', '<b>Color</b>'],
                  line_color='#eee5e5',
                  fill_color='#eee5e5',
                  align='left',
                  font=dict(size=12)),
      cells=dict(values=[heli, # 1st column
                        [f"{carac[h]['fuel_cap']}" for h in heli], #2nd colunm etc...
                        [carac[h]["init_fuel"] for h in heli], #3 ..
                        [carac[h]["start"] for h in heli],
                        [f"{carac[h]['conso_per_minute']} units/minute" for h in heli],
                        [f"{carac[h]['cost_per_min']} $/minute" for h in heli],
                        [f"{6} pers." for h in heli],
                        [len(served_by[h]) for h in heli],
                        ["" for h in heli]],
                line_color='#eee5e5',
                fill_color = col,
                #fill_color='#eee5e5',
                align='center'))
    ])
    fig.update_layout(paper_bgcolor="#eee5e5")

    return fig, imb

  @app.callback(
      [Output('failure-heli', 'options'),
      Output('failure-heli', 'value')],
      [Input('helicopter-selection', 'value')])
  def update_dropdown(value):
      dict_heli = {k : [f"h{i+1}" for i in range(k)] for k in range(2, 9)}
      def_index = 0
      return [[{'label':i, 'value':i} for i in dict_heli[value]], dict_heli[value][def_index]]



