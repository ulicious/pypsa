# coding: utf-8
"""
Adds extra extendable components to the clustered and simplified network.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        USD2013_to_EUR2013:
        dicountrate:
        emission_prices:

    electricity:
        max_hours:
        marginal_cost:
        capital_cost:
        extendable_carriers:
            StorageUnit:
            Store:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`

Inputs
------

- ``data/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.

Outputs
-------

- ``networks/{network}_s{simpl}_{clusters}_ec.nc``:


Description
-----------

The rule :mod:`add_extra_components` attaches additional extendable components to the clustered and simplified network. These can be configured in the ``config.yaml`` at ``electricity: extendable_carriers: ``. It processes ``networks/{network}_s{simpl}_{clusters}.nc`` to build ``networks/{network}_s{simpl}_{clusters}_ec.nc``, which in contrast to the former (depending on the configuration) contain with **zero** initial capacity

- ``StorageUnits`` of carrier 'H2' and/or 'battery'. If this option is chosen, every bus is given an extendable ``StorageUnit`` of the corresponding carrier. The energy and power capacities are linked through a parameter that specifies the energy capacity as maximum hours at full dispatch power and is configured in ``electricity: max_hours:``. This linkage leads to one investment variable per storage unit. The default ``max_hours`` lead to long-term hydrogen and short-term battery storage units.

- ``Stores`` of carrier 'H2' and/or 'battery' in combination with ``Links``. If this option is chosen, the script adds extra buses with corresponding carrier where energy ``Stores`` are attached and which are connected to the corresponding power buses via two links, one each for charging and discharging. This leads to three investment variables for the energy capacity, charging and discharging capacity of the storage unit.
"""
import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging

import pandas as pd
import geojson
from shapely.geometry import Point, Polygon, MultiPolygon, shape
import numpy as np
import pypsa
from pypsa.descriptors import Dict
from add_electricity import (load_costs, add_nice_carrier_names,
                             _add_missing_carriers_from_costs)

idx = pd.IndexSlice

def attach_storageunits(n, costs):
    elec_opts = snakemake.config['electricity']
    carriers = elec_opts['extendable_carriers']['StorageUnit']
    max_hours = elec_opts['max_hours']

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index

    for carrier in carriers:
        n.madd("StorageUnit", buses_i, ' ' + carrier,
               bus=buses_i,
               carrier=carrier,
               p_nom_extendable=True,
               capital_cost=costs.at[carrier, 'capital_cost'],
               marginal_cost=costs.at[carrier, 'marginal_cost'],
               efficiency_store=costs.at[carrier, 'efficiency'],
               efficiency_dispatch=costs.at[carrier, 'efficiency'],
               max_hours=max_hours[carrier],
               cyclic_state_of_charge=True)

def attach_stores(n, costs):
    elec_opts = snakemake.config['electricity']
    carriers = elec_opts['extendable_carriers']['Store']

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index
    bus_sub_dict = {k: n.buses[k].values for k in ['x', 'y', 'country']}

    if 'H2' in carriers:
        h2_buses_i = n.madd("Bus", buses_i + " H2", carrier="H2", **bus_sub_dict)

        n.madd("Store", h2_buses_i,
               bus=h2_buses_i,
               carrier='H2',
               e_nom_extendable=True,
               e_cyclic=True,
               capital_cost=costs.at["hydrogen storage", "capital_cost"])

        n.madd("Link", h2_buses_i + " Electrolysis",
               bus0=buses_i,
               bus1=h2_buses_i,
               carrier='H2 electrolysis',
               p_nom_extendable=True,
               efficiency=costs.at["electrolysis", "efficiency"],
               capital_cost=costs.at["electrolysis", "capital_cost"])

        n.madd("Link", h2_buses_i + " Fuel Cell",
               bus0=h2_buses_i,
               bus1=buses_i,
               carrier='H2 fuel cell',
               p_nom_extendable=True,
               efficiency=costs.at["fuel cell", "efficiency"],
               #NB: fixed cost is per MWel
               capital_cost=costs.at["fuel cell", "capital_cost"] * costs.at["fuel cell", "efficiency"])

    if 'battery' in carriers:
        b_buses_i = n.madd("Bus", buses_i + " battery", carrier="battery", **bus_sub_dict)

        n.madd("Store", b_buses_i,
               bus=b_buses_i,
               carrier='battery',
               e_cyclic=True,
               e_nom_extendable=True,
               capital_cost=costs.at['battery storage', 'capital_cost'])

        n.madd("Link", b_buses_i + " charger",
               bus0=buses_i,
               bus1=b_buses_i,
               carrier='battery charger',
               efficiency=costs.at['battery inverter', 'efficiency']**0.5,
               capital_cost=costs.at['battery inverter', 'capital_cost'],
               p_nom_extendable=True)

        n.madd("Link", b_buses_i + " discharger",
               bus0=b_buses_i,
               bus1=buses_i,
               carrier='battery discharger',
               efficiency=costs.at['battery inverter','efficiency']**0.5,
               capital_cost=costs.at['battery inverter', 'capital_cost'],
               p_nom_extendable=True)

def attach_hydrogen_pipelines(n, costs):
    elec_opts = snakemake.config['electricity']
    ext_carriers = elec_opts['extendable_carriers']
    as_stores = ext_carriers.get('Store', [])

    if 'H2 pipeline' not in ext_carriers.get('Link',[]): return

    assert 'H2' in as_stores, ("Attaching hydrogen pipelines requires hydrogen "
            "storage to be modelled as Store-Link-Bus combination. See "
            "`config.yaml` at `electricity: extendable_carriers: Store:`.")

    # determine bus pairs
    attrs = ["bus0","bus1","length"]
    candidates = pd.concat([n.lines[attrs], n.links.query('carrier=="DC"')[attrs]])\
                    .reset_index(drop=True)

    # remove bus pair duplicates regardless of order of bus0 and bus1
    h2_links = candidates[~pd.DataFrame(np.sort(candidates[['bus0', 'bus1']])).duplicated()]
    h2_links.index = h2_links.apply(lambda c: f"H2 pipeline {c.bus0}-{c.bus1}", axis=1)

    # add pipelines
    n.madd("Link",
           h2_links.index,
           bus0=h2_links.bus0.values + " H2",
           bus1=h2_links.bus1.values + " H2",
           p_min_pu=-1,
           p_nom_extendable=True,
           length=h2_links.length.values,
           capital_cost=costs.at['H2 pipeline','capital_cost']*h2_links.length,
           efficiency=costs.at['H2 pipeline','efficiency'],
           carrier="H2 pipeline")

def add_other_carriers(network):
    
    #See in config file, which other carriers should be added and add them to network
    #Similar to carriers like nuclear etc.
    carrier_query = snakemake.config['other_carriers'] #Choose considered carriers
    
    if not carrier_query:
        logger_info("No carriers chosen")
    else:    
        for c in carrier_query:

            network.add('Carrier',
                        c)
            logger.info("Carrier {} was added".format(c))

def add_new_components(snakemake=None):

#This methods adds the new components, based on the config file

    override_components = pypsa.components.components.copy()
    override_components_attrs = Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})

    #Check the ___ csv and add the components to the network, if the carrier is chosen in the config file. Furthermore, it can be chosen, which
    #type of facility is added
    component_query = snakemake.config['other_components']['source']

    #CSV file with all components, which can be added with the config file
    file_path = "~/PyPSA/pypsa-eur/data/new_components/new_components_list.csv"
    df_new_components = pd.read_csv(file_path, index_col=False)

    if not component_query:
        logger.info("No new components chosen")
        new_components = False
    else:
        new_components = True
        for c in component_query:

            components_list = df_new_components['nice_name'].tolist()

            if not c in components_list: #Check if components in config are also in the csv file
                logger.warning('Component ' + c + ' not in csv list. Please add')
            else:

                index = df_new_components.index[df_new_components['nice_name'] == c].tolist()

                #Name, Description, type of new component
                override_components.loc[c] = [df_new_components.loc[index, 'list_name_component'].values[0],
                                              df_new_components.loc[index, 'description_component'].values[0],
                                              df_new_components.loc[index, 'type_component'].values[0]]

                #Load corresponding csv file of new component
                file_path = "~/PyPSA/pypsa-eur/data/new_components/{}.csv".format(c)
                df_new_component_attributes = pd.read_csv(file_path)

                #Create list with attributes of current component
                attributes = df_new_component_attributes['attribute'].tolist()

                #Connect the attributes dataframe to the override attrs dataframe
                override_components_attrs[c] = df_new_component_attributes
                override_components_attrs[c].set_index('attribute',inplace=True)

                logger.info("Component " + c + " was added to the components list")
                
    return override_components, override_components_attrs


def attach_facilities(network, regions_onshore):

    def bus_allocation(facility_df, regions_onshore):
        #Uses the GEOJSON file to allocate the facilities to the voronoi cells

        with open(regions_onshore) as f:
            regions = geojson.load(f)

        for i in range(0,len(facility_df.index)):
            point = Point((facility_df.iloc[i]['lon'], facility_df.iloc[i]['lat']))

            for feature in regions['features']:
                poly = shape(feature['geometry'])

                if point.within(poly):
                    facility_df.loc[i,'bus'] = feature['properties']['name']
                    break
    
    component_query = snakemake.config['other_components']['source']

    for c in component_query:

        #Load file with different facilities and adjust bus regarding regions_onshore
        facility_df = pd.read_csv("~/PyPSA/pypsa-eur/data/new_components/{}_facilities.csv".format(c), index_col=False)
        facility_df['bus'] = np.nan
        bus_allocation(facility_df, regions_onshore)

        #Check start and end date in config file
        start_date = snakemake.config['snapshots']['start']
        end_date = snakemake.config['snapshots']['end']
        
        try:
            #Load profile csv. Important: Similar to load demand, profile has to have a value for each snapshot
            profile_file = pd.read_csv("~/PyPSA/pypsa-eur/data/new_components/{}_profiles.csv".format(c), index_col='id')
            profile_exists = True
        except:
            logger.info('No profile for component ' + c + ' exists. Continuing without')
            profile_exists = False

        #rename facilities for better identification
        facility_df = facility_df.rename(index=lambda s: c + "_" + str(s))
        
        #iterate through facility csv and add them to the network
        for index in facility_df.index:

            list_name = [index]

            if profile_exists:
            
                #get current facility ID
                identification = str(facility_df.loc[index,'id'])
                
                #get profile of facility based on ID from profile csv
                profile = pd.DataFrame(profile_file.loc[:,identification])

                #Create DateTimeIndex for utc timestamp of profile
                times = pd.DataFrame(pd.DatetimeIndex(profile.index.values),
                                     columns=['utc_timestamp'])

                #Set utc timestamp as index
                profile.set_index(times['utc_timestamp'],inplace=True)
                profile[identification] = profile[identification].astype(float)
                profile.rename(columns={identification:index}, inplace=True)

                #Check, whether snapshot timestamps equal each other
                if profile.index.values[0] < pd.Timestamp(start_date) or profile.index.values[-1] > pd.Timestamp(end_date):
                    raise IndexError('Profile not within the snapshot range. Please adjust either config file or corresponding profile csv')

                #Add facility and profile to network
                n.madd(c, names=list_name,
                       name=facility_df.loc[index, 'Name'],
                       resource=facility_df.loc[index, 'Facility'],
                       profile=profile,
                       bus=facility_df.loc[index,'bus'])
            else:
                #add facility to network
                n.madd(c, names=list_name,
                       name=facility_df.loc[index, 'Name'],
                       resource=facility_df.loc[index, 'Facility'],
                       bus=facility_df.loc[index,'bus'])

def profile_aggregation(network):

    component_query = snakemake.config['other_components']['source']

    for c in component_query:

        components_file = pd.read_csv("~/PyPSA/pypsa-eur/data/components/new_components_list.csv", index_col=0)

        aggregation_type = components_file.loc[c][aggregation_type]
        #Three different aggregation type: None, temporal, spatial, temporal and spatial

        logger.info('Resampling the network to {}'.format(offset))
        m = n.copy(with_time=False)

        snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
        m.set_snapshots(snapshot_weightings.index)
        m.snapshot_weightings = snapshot_weightings

        for nc in n.iterate_components():
            if nc == c:
                pnl = getattr(m, c.list_name+"_t")
                for k, df in iteritems(c.pnl):
                    if not df.empty:
                        pnl[k] = df.resample(offset).sum()

    return m


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('add_extra_components', network='elec',
                                  simpl='', clusters=5)
    configure_logging(snakemake)

    #--------------
    other_components = snakemake.config['other_components']['source']

    if other_components is not None:
        logger.info("New components detected. Network will be build using component overriding method")

        override_components, override_components_attrs = add_new_components(snakemake)
        
        n = pypsa.Network(snakemake.input.network,
                          override_components=override_components,
                          override_component_attrs=override_components_attrs)
        n.name = 'PyPSA-Eur'
    else:
        n = pypsa.Network(snakemake.input.network)
        n.name = 'PyPSA-Eur'
    #---------------
    


    Nyears = n.snapshot_weightings.sum()/8760.
    costs = load_costs(Nyears, tech_costs=snakemake.input.tech_costs,
                       config=snakemake.config['costs'],
                       elec_config=snakemake.config['electricity'])
    regions_onshore = snakemake.input.regions_onshore

    attach_storageunits(n, costs)
    attach_stores(n, costs)
    attach_hydrogen_pipelines(n, costs)

    add_nice_carrier_names(n, config=snakemake.config)

    add_other_carriers(n)
    attach_facilities(n, regions_onshore)

    n.export_to_netcdf(snakemake.output[0])
