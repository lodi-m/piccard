import warnings
import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import re

def preprocessing(ct_data, year, id):
  '''
  Return a cleaned geopandas df of the inputted CT data.
  Note: Input data is assumed to have been passed through gpd.read_file()
  '''
  process_data = ct_data.copy()

  #Suppressing CRS warning associated with .buffer()
  with warnings.catch_warnings():
      warnings.simplefilter(action='ignore', category=UserWarning)
      process_data['geometry'] = (process_data.to_crs('EPSG:4246').geometry
                                  .buffer(-0.000001))
      process_data['area' + '_' + year] = process_data.area
  process_data[id] = year + '_' + process_data[id]

  return process_data


def ct_containment(preprocessed_dfs, years):
  '''
  Returns a GeoDataFrame with census tracts which are contained
  within a census tract from the following census
  '''
  num_years = len(years)
  contained_tracts = []

  for i in range(num_years-1):
      #Getting CTs which are contained within a previous year's CT
      contained_df = gpd.overlay(preprocessed_dfs[i], preprocessed_dfs[i+1],
                                  how='intersection')
      with warnings.catch_warnings():
          warnings.simplefilter(action='ignore', category=UserWarning)

          contained_df['area_intersection'] = contained_df.area
          #Calculating the percentage of the overlapping area between the 2 years
          pct_col = 'pct_' + years[i+1] + '_of_' + years[i]
          contained_df[pct_col] = (contained_df['area_intersection'] /
                                    contained_df[['area_'+years[i],
                                                  'area_'+years[i+1]]].min(axis=1))
      contained_tracts.append(contained_df)
  return contained_tracts


def get_nodes(contained_tracts_df, id, threshold=0.05):
  '''
  Returns a GeoDataFrame with the graph connections between two census tracts
  of different years. Each row corresponds to one edge in the final network.
  '''
  nodes = gpd.GeoDataFrame()
  id_cols = [f'{id}_1', f'{id}_2']

  #Aggregating overlapped percentage area for all unique CTs
  for i in range(len(contained_tracts_df)):
      pct_col = contained_tracts_df[i].iloc[:, -1].name
      year_pct = (contained_tracts_df[i]
                  .groupby(id_cols)
                  .agg({f'{pct_col}': 'sum'})
                  .reset_index()
                  )

      #Selecting CTs with an overlapped area above user's threshold
      connected_cts = year_pct[year_pct[pct_col] >= threshold][id_cols]
      nodes = pd.concat([nodes, connected_cts], axis=0, ignore_index=True)

  return nodes


def assign_node_level(row, years, id):
  """
  Assigns the level of a node in the network based on its relative year in the
  network
  Example: All 2021 nodes are in level 3 in a graph with years 2011, 2016, 2021
  """
  for i in range(len(years)):
    if row[id].startswith(str(years[i])):
      return i+1
    

def get_attributes(nodes, census_dfs, years, id):
  '''
  Returns all the attributes in the original data corresponding to the network
  nodes
  '''
  #Condensing nodes into single column df
  single_nodes = pd.concat([nodes[col] for col in nodes]).reset_index(drop=True)
  single_nodes_df = pd.DataFrame({id: single_nodes})
  attr = []

  for i in range(len(census_dfs)):
      #Adding year as a prefix for the merge
      curr_df_id = census_dfs[i].loc[:, id]
      curr_df_id = years[i] + '_' + curr_df_id

      #Removing geometry column in attributes for the final table
      year_attr = census_dfs[i].loc[:, (census_dfs[i].columns != 'geometry')].copy()
      year_attr[id] = curr_df_id
      year_attr = pd.merge(single_nodes_df, year_attr, on=id, how='right')

      attr.append(year_attr)
  all_attr = (pd.concat(attr)).drop_duplicates(subset=id)
  all_attr = all_attr[all_attr[id].notna()]

  #Assigning each node it's level in the network (used for mainly drawing)
  all_attr['network_level'] = all_attr.apply(lambda x: assign_node_level(x, years, id), axis=1)
  return all_attr


def create_network(census_dfs, years, id, threshold=0.05):
  '''
  Create network corresponding to input nodes and attributes.
  '''
  preprocessed_dfs = [preprocessing(census_dfs[i], years[i], id) for i in range(len(census_dfs))]
  contained_cts = ct_containment(preprocessed_dfs, years)

  nodes = get_nodes(contained_cts, id, threshold)
  attributes = get_attributes(nodes, census_dfs, years, id)

  G = nx.from_pandas_edgelist(nodes, f'{id}_1', f'{id}_2')
  nx.set_node_attributes(G, attributes.set_index(id).to_dict('index'))

  return G


def find_all_paths(nodes_df, num_joins, id):
  '''
  Return all possible paths present in the input data.
  Note: The resulting dataframe is not organized and does contain
        duplicate entries in both the rows and columns.
  '''
  left_cols = [f'{id}_1_x', f'{id}_2_x']
  right_cols = [f'{id}_1_y', f'{id}_1_x']

  #Merging network nodes num_joins amount of times to ensure all paths are found
  curr_join = nodes_df.merge(nodes_df, how='left', left_on=f'{id}_1', right_on=f'{id}_2')
  curr_join = curr_join.sort_values(by=[f'{id}_1_y', f'{id}_2_y'], ignore_index=True)

  if num_joins > 1:
      for i in range(num_joins - 1):
          curr_join = curr_join.merge(curr_join, how='left', left_on=left_cols, right_on=right_cols, suffixes=['x', 'y'])
          #Accounting for the new column names after the merge
          left_cols = [col_name + 'x' for col_name in left_cols]
          right_cols = [col_name + 'x' for col_name in right_cols]
  return (curr_join, left_cols, right_cols)


def find_full_paths(full_paths_df, final_cols):
  '''
  Return all full paths present in input data.
  Note: Define a full path as a path in the network where the starting node is
        from the first input year and the ending node is from the last input year.
  '''
  full_paths = pd.DataFrame()

  if (not full_paths_df.empty):
      full_paths = full_paths_df.T.drop_duplicates().sort_values(by=0).T
      full_paths.columns = final_cols
  return full_paths


def first_year_partial_paths(all_partial_paths, years, final_cols):
  '''
  Return all partial paths only for the first input year.
  Note: Define a partial path as a path in the network where the starting and
        ending nodes are of any year (i.e., not a full path).
  '''
  num_years = len(years)
  drop_cols = final_cols[1:]

  #Selecting paths with the starting node as the first year
  mask = all_partial_paths.iloc[:, 0].str.startswith(years[0] + '_')
  first_year_partials = all_partial_paths[mask]

  #Checking if df empty or not
  if len(first_year_partials.index) != 0:
    #Calculating which year contains the ending node
    max_partial_year = max(all_partial_paths.T.stack().values)[:4]

    #Appending NaN columns to the end for each year as they don't exist in data
    if ((max_partial_year >= years[1]) & (max_partial_year != years[-1])):
        for i in reversed(range((num_years - 1) - max_partial_year)):
            last_col = len(first_year_partials.columns)
            first_year_partials.insert(last_col, final_cols[-i], np.NaN)
        first_year_partials.columns = final_cols
    first_year_partials = first_year_partials.T.drop_duplicates().dropna().T
    first_year_partials.columns = final_cols
    return first_year_partials
  else:
    empty_df = pd.DataFrame(columns = final_cols)
    return empty_df
  

def unique_partial_paths(all_partial_paths, years, left_cols, final_cols):
  '''
  Return all unique partial paths between two consecutive input years.
  Note: Define a partial path as a path in the network where the starting and
        ending nodes are of any year (i.e., not a full path).
  '''
  num_years = len(years)
  unique_partials = pd.DataFrame()

  for i in range(1, num_years):
      curr_year = years[i] + '_'
      prev_year = years[i-1] + '_'

      curr_year_mask = all_partial_paths.iloc[:, 0].str.startswith(curr_year)
      prev_year_mask = all_partial_paths.iloc[:, 0].str.startswith(prev_year)

      curr_year_partials = all_partial_paths[curr_year_mask]
      prev_year_partials = all_partial_paths[prev_year_mask]

      curr_year_mask = ~curr_year_partials[left_cols[0]].isin(prev_year_partials)
      curr_year_unique = curr_year_partials[curr_year_mask]
      curr_year_unique = curr_year_partials.dropna(axis=1).T.drop_duplicates().T

  #Appending NaN column to the front to account for missing first year
      for k in range(i):
          curr_year_unique.insert(0, final_cols[k], np.NaN)

  #Appending NaN column to the end to account for missing last year
      if(not curr_year_unique.empty):
          curr_year_val = max(curr_year_unique.T.stack().values)[:4]
          curr_year_index = years.index(curr_year_val)

          if (curr_year_index != years[-1]):
              for j in range((num_years - 1) - curr_year_index):
                  last_col = len(curr_year_unique.columns)
                  curr_year_unique.insert(last_col, final_cols[-j], np.NaN)

          curr_year_unique.columns = final_cols
      unique_partials = pd.concat([unique_partials, curr_year_unique])
  return unique_partials


def find_partial_paths(partial_paths_df, years, left_cols, final_cols, exclude_nodes):
  '''
  Return all partial paths present in input data.
  Note: Define a partial path as a path in the network where the starting and
        ending nodes are of any year (i.e., not a full path).
  '''

  all_partial_paths = partial_paths_df.T.drop_duplicates().T
  all_partial_paths = all_partial_paths[~all_partial_paths[left_cols[0]].isin(exclude_nodes)]

  first_year_partials = first_year_partial_paths(all_partial_paths, years, final_cols)
  unique_partials = unique_partial_paths(all_partial_paths, years, left_cols, final_cols)
  all_partials = pd.concat([unique_partials, first_year_partials])

  return all_partials


def attach_attributes(network_table, attributes, years, final_cols, id):
  '''
  Return network table with attached attributes corresponding to the nodes
  involved.
  '''
  years_df_list = []

  for i in range(len(final_cols)):
      col = str(final_cols[i])

      #Getting attributes for each year
      table_col = network_table[col].to_frame().astype(object)
      curr_year = table_col.merge(attributes, how='left', left_on=col, right_on=id)
      curr_year = curr_year.drop([id], axis=1)

      #Suppressing warning for str.replace
      with warnings.catch_warnings():
          warnings.simplefilter(action='ignore', category=FutureWarning)
          curr_year = curr_year.apply(lambda x: x.str.replace(r'[0-9]+_', '') if x.dtypes==object else x).reset_index(drop=True)

          #Formatting all columns as 'colname_year'
          curr_year_cols = [f'{col}_{years[i]}' if col != final_cols[i] and col != f'area_{years[i]}' else col for col in curr_year.columns]
          curr_year.columns = curr_year_cols
          years_df_list.append(curr_year)

  #Combining all years dfs into one
  network_table = (pd.concat(years_df_list, axis=1)).dropna(how='all', axis=1)
  return network_table


def create_network_table(census_dfs, years, id, threshold=0.05):
  '''
  Return the final network table with all the temporal connections present in
  the input data over all the input years.
  '''
  num_years = len(years)
  num_joins = math.ceil(num_years/2)
  final_cols = [id + '_' + col_name for col_name in years]
  network_table = pd.DataFrame()
  drop_cols = final_cols[1:]

  preprocessed_dfs = [preprocessing(census_dfs[i], years[i], id) for i in range(len(census_dfs))]
  contained_cts = ct_containment(preprocessed_dfs, years)
  nodes = get_nodes(contained_cts, id, threshold)

  #all_paths returns a three item tuple
  all_paths = find_all_paths(nodes, num_joins, id)
  all_paths_df = all_paths[0]
  left_cols = all_paths[1]
  right_cols = all_paths[2]

  #Dividing all network paths into full paths and partial paths
  na_df = all_paths_df[all_paths_df.isnull().any(axis=1)]
  no_na_df = all_paths_df[~all_paths_df.isnull().any(axis=1)]

  full_paths = find_full_paths(no_na_df, final_cols)
  full_paths_list = full_paths.to_numpy().flatten()

  partial_paths = find_partial_paths(na_df, years, left_cols, final_cols, full_paths_list)

  network_table = pd.concat([full_paths, partial_paths])
  network_table = network_table[final_cols]
  network_table = network_table.T.drop_duplicates().T
  network_table = network_table.drop_duplicates(subset=drop_cols, keep='last')
  network_table.sort_values(by=final_cols[0], ignore_index=True)

  attributes = get_attributes(nodes, census_dfs, years, id)
  final_table = attach_attributes(network_table, attributes, years, final_cols, id)

  #Formatting final table columns
  for i in range(len(final_cols)):
      col = str(final_cols[i])
      popped = final_table.pop(col)
      final_table.insert(i, popped.name, popped)
  final_table.columns= final_table.columns.str.lower()

  return final_table


def draw_subnetwork(network_table, G, sample_pct=0.005):
  """
  Draws a subgraph of the network representation, using a sample_pct percentage
  path sample from the network table.
  """
  r = re.compile('ctuid_[0-9]+')
  table_cols = list(network_table.columns)
  sample_cols = list(filter(r.match, table_cols))

  network_table = network_table[sample_cols]
  sample_table = network_table.sample(frac=sample_pct)

  #Adding corresponding year prefix to all nodes
  for i in range(len(sample_cols)):
      curr_col = sample_cols[i]
      sample_table[curr_col] = sample_cols[i][6:] + '_' + sample_table[curr_col]
  sample_nodes = sample_table.stack().droplevel(1).sort_values()

  subgraph = G.subgraph(sample_nodes)

  plt.figure(figsize=(20,30))
  pos = nx.multipartite_layout(subgraph, subset_key='network_level')

  nx.draw(subgraph, pos, with_labels=True)
  plt.show()


