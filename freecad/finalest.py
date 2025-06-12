from shapely.affinity import scale, translate
from shapely.geometry import box, Point
from shapely.ops import nearest_points
import FreeCAD as App
import re
import math
import datetime
from tkinter import *
from tkcalendar import Calendar
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.cm as cm
import Part
from collections import defaultdict
import pandas as pd
import Draft  # For draft tools like adding blocks
from shapely.geometry import Polygon, Point, box
from shapely.affinity import translate, rotate
import string
import plotly.express as px
import os
import FreeCADGui
from PIL import Image
import io
from PySide2 import QtWidgets, QtCore

def create_gantt_chart(project_data):
    df = pd.DataFrame(project_data)
    fig = px.timeline(df, x_start="start", x_end="lifting", y="name", color="module", hover_data=["zone"])
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(template='plotly_white')
    fig.write_html('figure.html', auto_open=False)
    fig.show()


font_path = r"C:\Users\amrin.kareem\Downloads\work\NMDC-Team-Work\YardLayout\consolas\consola.ttf"

def create_project_block(project_name, module, lifting, polygon, proj, color = [255, 255, 0]):
    """
    Creates a block (polygon) in FreeCAD with the given name, coordinates, and color.

    :param project_name: Name of the project (used as the block's name in FreeCAD)
    :param polygon_coords: List of tuples representing the polygon vertices [(x1, y1), (x2, y2), ...]
    :param color_code: List of three integers representing the RGB color code of the block.
    :return: Part::Feature object of the created block.
    """
    doc = App.ActiveDocument
    polygon_coords = list(polygon.exterior.coords)  # Get list of (x, y) tuples
    # Convert tuple coordinates to FreeCAD vectors
    polygon_vectors = [App.Vector(x, y, 0) for x, y in polygon_coords]

    # Ensure the polygon is closed (last point should match the first)
    if polygon_vectors[0] != polygon_vectors[-1]:
        polygon_vectors.append(polygon_vectors[0])
    
    # Create a wire representing the polygon
    polygon_shape = Part.makePolygon(polygon_vectors)
    polygon_face = Part.Face(polygon_shape)
    
	# Create the shape and a part object for the polygon
    block = doc.addObject("Part::Feature", project_name + "\n" + module)
    block.Shape = polygon_face

        # Add custom property to identify macro-created objects
    if not hasattr(block, "macroCreated"):
        block.addProperty("App::PropertyBool", "macroCreated", "Macro")
    block.macroCreated = True

    # Access the shape from the block
    vertices = block.Shape.Vertexes

    # Extract coordinates
    x_coords = [v.Point.x for v in vertices]
    y_coords = [v.Point.y for v in vertices]

    # Calculate width and height using parallel edges 
    edges = []
    for i in range(len(x_coords)-1):
        edge_length = math.sqrt((x_coords[i+1]-x_coords[i])**2 + (y_coords[i+1]-y_coords[i])**2)
        angle = math.atan2(y_coords[i+1]-y_coords[i], x_coords[i+1]-x_coords[i])
        edges.append((edge_length, angle))
    
    # Get the indices of the two edges, long and short
    shortest_edge = sorted(edges, key=lambda x: x[0])[0]
    longest_edge = sorted(edges, key=lambda x: x[0], reverse=True)[0]
    width = shortest_edge[0]
    height = longest_edge[0]  
    angle = longest_edge[1]  
    angle_degrees = math.degrees(angle)

		
	# Compute centroid of the polygon (Shapely)
    centroid = polygon.centroid
    centroid_vector = App.Vector(centroid.x, centroid.y, 0)

	# Calculate scaling factors to fit text within module dimensions
    max_text_width = width - 2000  # Leave some margin
    max_text_height = height / 3


    def get_text_width(text, font_path, height):
        temp_text = Draft.make_shapestring(text, font_path, 2000, height)
        text_width = temp_text.Shape.BoundBox.XLength
        App.ActiveDocument.removeObject(temp_text.Name)
        return text_width

    def color_name_to_rgb(name):
        try:
            rgb = mcolors.to_rgb(name)  # returns (r, g, b) each in range [0, 1]
            return tuple(int(c * 255) for c in rgb)  # convert to 0–255 range
        except ValueError:
            print(f"Warning: Invalid color name '{name}', defaulting to gray.")
            return (128, 128, 128)  # fallback colo

	# Calculate scale for each text line
    scale_factors = [max_text_width / get_text_width(text, font_path, 5) for text in [project_name, module, str(lifting.date())]]
    scale_factor_height = max_text_height
    scale_factor = min(min(scale_factors), scale_factor_height)

	# Offset text vertically (stack one above the other)
    text_spacing = 2000  # Spacing between lines
    min_x = min(p.x for p in polygon_vectors)  # Left edge
    centroid_y = polygon.centroid.y  # Middle Y position

    # Create first line of text (platfrom_name)
    text1 = Draft.make_shapestring(project_name, font_path, 2000*scale_factor, 5*scale_factor)
    text1.Placement.Base = App.Vector(min_x + 1000, centroid_y + text_spacing*scale_factor, 0)  # Shift up
    # Apply rotation to the text
    exclusion = [angle + offset for angle in [0, 90, 180, 270, 360] for offset in (-5, 5)]
    if angle_degrees in exclusion:
        text1.Placement.Rotation = App.Rotation(App.Vector(0, 0, 1), angle_degrees)

    # Create second line of text (module)
    text2 = Draft.make_shapestring(module, font_path, 2000*scale_factor, 5*scale_factor)
    text2.Placement.Base = App.Vector(min_x + 1000, centroid_y - text_spacing*scale_factor, 0)  # Shift down
    
    # Create third line of text (lifting date)
    text3 = Draft.make_shapestring(str(lifting.date()), font_path, 2000*scale_factor, 5*scale_factor)
    text3.Placement.Base = App.Vector(min_x + 1000, centroid_y - 3*text_spacing*scale_factor, 0)  # Shift down

    # Create fourth line of text (project)
    text4 = Draft.make_shapestring(str(proj), font_path, 2000*scale_factor, 5*scale_factor)
    text4.Placement.Base = App.Vector(min_x + 1000, centroid_y - 6*text_spacing*scale_factor, 0)  # Shift down

    # Add custom property to text objects as well
    for text_obj in (text1, text2, text3, text4):
        if not hasattr(text_obj, "macroCreated"):
            text_obj.addProperty("App::PropertyBool", "macroCreated", "Macro")
        text_obj.macroCreated = True
    
	# Set block color
    color = color_name_to_rgb(color) 
    block.ViewObject.ShapeColor = tuple(c / 255.0 for c in color)
    block.ViewObject.DiffuseColor = tuple(c / 255.0 for c in color)
    # Recompute the document to update the view
    doc.recompute()
    
    return block

def place_blocks_in_free_space(blocks_data):
    """
    Place blocks within a given free space polygon, avoiding overlap.
    
    :param free_space_polygon: A Shapely Polygon object representing the free space area.
    :param blocks_data: List of dictionaries containing block data (coords, dimensions, color).
    :return: List of placed blocks in FreeCAD.
    """
    placed_blocks = []

    for block_data in blocks_data:
        # Extract project information
        project_name = block_data["name"]
        module = block_data['module']
        coords = block_data["coords"]
        ldate = block_data["lifting"]
        color = block_data["color"]  
        proj = block_data["proj"]
        # Create a block (rectangle)
        create_project_block(project_name, module, ldate, coords, proj, color)
        

def compute_free_spaces_from_poly_names(target_names):

    """
    Extracts only the polygons (Draft Wires) whose names are in the given target_names list.
    
    :param target_names: List of polygon names to extract
    :return: Dictionary with polygon names as keys and Shapely Polygons as values
    """
    doc = App.ActiveDocument
    free_spaces = {}

    for obj in doc.Objects:
        if obj.Label in target_names and hasattr(obj, "Shape"):
            shape = obj.Shape
            # Convert FreeCAD vertices to points
            polygon_points = [(v.X, v.Y) for v in shape.Vertexes]
            
            # Create a Shapely polygon and store it in the dictionary
            free_spaces[obj.Label] = Polygon(polygon_points)

    return free_spaces


def get_df(date_to_view, file): # Example blocks data (similar to your dataframe)
    def create_platforms(west_oil_platform_names, east_oil_platform_names, module_names, default_status):
        platforms = {"PDM": {},
                    "JACKETS": {},
                    "SSVS:": {},
                    "DECKS": {},
                    "RETROFIT": {}}  
        
        for platform1, platform2 in zip(west_oil_platform_names, east_oil_platform_names):
            platforms["PDM"][platform1] = {"modules": {}}
            platforms["PDM"][platform2] = {"modules": {}}
            for module in module_names:
                platforms["PDM"][platform1]["modules"][module] = default_status.copy()
                platforms["PDM"][platform2]["modules"][module] = default_status.copy()
        
        return platforms

    # List of platforms
    west_oil_platform_names = [
        "ZULF 1230/1239", "ZULF 1220/1229", "ZULF 1250/1259", "ZULF 1400/1409", 
        "ZULF 1210/1219", "ZULF 1260/1269", "RBYN 10/19", "ZULF 1200/1209", 
        "ZULF 1240/1249", "ZULF 1270/1279", "ZULF 1280/1289", "ZULF 1490/1499", 
        "ZULF 1390/1399"
    ]
    east_oil_platform_names = ['ZULF 1330/1339', 'ZULF 1360/1369', 'ZULF 1370/1379',
        'ZULF 1290/1299', 'ZULF 1530/1539', 'ZULF 1340/1349',
        'ZULF 1380/1389', 'ZULF 1410/1419', 'ZULF 1300/1309',
        'ZULF 1310/1319', 'ZULF 1320/1329']
    # List of module names
    module_names = [
        "MAIN DECK", "MEZZANINE DECK", "HELIDECK DECK", "WELLHEAD ACCESS PLATFORM",
        "CELLAR DECK", "FULL DECK"
    ]

    # Default status dictionary
    default_status = {
        "START": "",
        "LIFTING": ""
    }

    # Add additional status for FULL DECK
    full_deck_status = {
        "START": "",
        "LOADOUT_READY": "15-Sep-2023",
        "LOADOUT": "15-Sep-2023"
    }

    # Generate platforms dictionary
    platforms = create_platforms(west_oil_platform_names, east_oil_platform_names, module_names, default_status)

    # Assign specific statuses to FULL DECK
    for platform in platforms["PDM"]:
        platforms["PDM"][platform]["modules"]["FULL DECK"] = full_deck_status.copy()

    
    xls = pd.ExcelFile(file)
    df1 = pd.read_excel(xls, sheet_name=0, header=1, parse_dates=['Start', 'Finish'])
    df2 = pd.read_excel(xls, sheet_name=1, header=1, parse_dates=['Start', 'Finish'])
    for df in [df1, df2]:
        df['Project'] = df.loc[0, 'Activity ID']
        df['Title'] = df.loc[4, 'Activity ID'] 
        df['Main Module'] = df['Activity ID'].where((df['Activity Name'].isna()) & (df['Activity ID'].str.contains('PDM DECK') | df['Activity ID'].str.contains('JACKET')))
        df['Main Module'] = df['Main Module'].fillna(method='ffill') 
        df['SubModule'] = df['Activity ID'].where((df['Activity Name'].shift(-1).notna())  & (df['Activity Name'].isna()) & (df['Title'].str.contains('PDM')))
        df['SubModule'] = df['SubModule'].fillna(method='ffill') 
        df['Start'] = df['Start'].apply(pd.to_datetime, errors = 'coerce')
        df['Finish'] = df['Finish'].apply(pd.to_datetime, errors = 'coerce')
    

    submodule_map = {
        "ERECTION MAIN DECK": "MAIN DECK",
        "ERECTION CELL AR DECK": "CELLAR DECK",
        "ERECTION WELLHEAD ACCESS PLATFORM": "WELLHEAD ACCESS PLATFORM",
        "ERECTION MEZZANINE DECK": "MEZZANINE DECK",
        "ERECTION HELIDECK DECK": "HELIDECK DECK",
        "LOAD OUT & SEA FASTENING": "FULL DECK",
        "ASSEMBLY & WELD MAIN DECK": "MAIN DECK",
        "ASSEMBLY & WELD CELLAR DECK": "CELLAR DECK",
        "ASSEMBLY & WELD WELLHEAD ACCESS PLATFORM": "WELLHEAD ACCESS PLATFORM",
        "ASSEMBLY & WELD MEZZANINE DECK": "MEZZANINE DECK",
        "ASSEMBLY & WELD HELIDECK DECK": "HELIDECK DECK"
    }

    dimensions = {
        "MAIN DECK": (22000, 38000), 
        "CELLAR DECK": (14000, 18000),
        "WELLHEAD ACCESS PLATFORM": (13000, 21000),
        "MEZZANINE DECK": (22000, 33000), 
        "HELIDECK DECK": (22000, 30000),
        "FULL DECK": (23000, 45000)
    }

    # Define corresponding values for each condition
    choices = ['SUBSEA VALVE SKIDS', 'RETROFIT RISERS', 'JACKETS', 'DECK', 'PDM']

    
    for df in [df1, df2]:
        df['Starting Assembly'] = df["Start"].where(df['Activity Name'].str.contains('FIT-UP & TACK WELD MAIN FRAME', na = False, case=False))
        df["Lifting_Date"] = df["Start"].where(df['Activity Name'].str.contains('LIFT & INSTALL', na=False))
        df["Loadout_Date"] = df["Finish"].where(df['Activity Name'].str.contains('READY FOR LOAD OUT', na=False) | df['Activity Name'].str.contains('READY FOR L/O', na=False) | df['Activity Name'].str.contains('READY FOR LOADOUT', na=False))
        df["Seafastening"] = df["Start"].where(df['Activity Name'].str.contains('LOAD OUT AND SEA FASTENING', na=False))
        df['dimensions'] = df['SubModule'].map(dimensions)
        df["SubModule"] = df["SubModule"].replace(submodule_map)
        
    df = pd.concat([df1, df2])
    df = df.drop(['Unnamed: 5', 'Remaining Duration', 'Activity ID'], axis=1)  
    
    for index, row in df.iterrows():
        if pd.notna(row['Seafastening']) and "DECK" in row['Activity Name'] and all(substring not in row['Activity Name'] for substring in ["TP-", "TP-3", "EDP-2"]):
            df.at[index, 'SubModule'] = "FULL DECK"
            a = re.sub(r':-|:', '', ' '.join(row['Activity Name'].split(' ')[:4]))
            df.at[index, 'Main Module'] = ' '.join(a.split()) 
            
    for index, row in df.iterrows():
        for j in platforms["PDM"].keys():
            for i in ["MAIN DECK", "MEZZANINE DECK", "HELIDECK DECK", "WELLHEAD ACCESS PLATFORM", "CELLAR DECK", "FULL DECK"]:
                    # Filter the DataFrame to find rows matching the current 'SubModule'
                    
                if i != "FULL DECK":
                        if row['SubModule'] == i and j in str(row['Main Module']):
                            if pd.notna(row['Lifting_Date']) and "LIFT & INSTALL" in row['Activity Name']:
                                
                                platforms["PDM"][j]["modules"][i]["LIFTING"] = row['Lifting_Date'] # The Lifting Date is the date when the decks disappear from their assigned area
                            if pd.notna(row['Starting Assembly']):
                                if i == "MAIN DECK":
                                    
                                    platforms["PDM"][j]["modules"][i]["START"] = row['Starting Assembly'] - pd.Timedelta(days=8) # Main deck assembly starts 8 days before given main deck start date
                                else:
                                    platforms["PDM"][j]["modules"][i]["START"] = row['Starting Assembly']
                                    
            if row['SubModule'] == "FULL DECK" and j in str(row['Main Module']):
                start_date = platforms["PDM"][j]["modules"]["CELLAR DECK"]["START"]
                platforms["PDM"][j]["modules"]["FULL DECK"]["START"] = start_date
                if pd.notna(row['Loadout_Date']):
                    platforms["PDM"][j]["modules"]["FULL DECK"]["LOADOUT_READY"] = row['Loadout_Date']
                if pd.notna(row['Seafastening']):
                    platforms["PDM"][j]["modules"]["FULL DECK"]["LOADOUT"] = row['Seafastening']

    # Initialize an empty list to store flattened data
    flattened_data = []

    # Iterate through the nested dictionary
    for platform, platform_data in platforms['PDM'].items():
        for module, module_data in platform_data['modules'].items():
            row = {
                'Platform': platform,
                'Module': module,
                **module_data  # Unpack the module data (START, LIFTING, etc.)
            }
            flattened_data.append(row)

    # Create a DataFrame
    df_pdm = pd.DataFrame(flattened_data)
    df_pdm.columns = ['Platform', 'Module', 'Start', 'Lifting', 'Loadout_Ready', 'Loadout']
    df_full_deck_pdm = df_pdm[df_pdm['Module'] == 'FULL DECK'][['Platform', 'Module', 'Start', 'Loadout_Ready', 'Loadout']]
    df_pdm = df_pdm[df_pdm['Module'] != 'FULL DECK'] 
    df_pdm = df_pdm.drop(columns=['Loadout_Ready', 'Loadout'], axis=1)  # Drop specific columns
    for df in [df_full_deck_pdm, df_pdm]:
        df['dimensions'] = df['Module'].map(dimensions)
        
    # month_num = datetime.datetime.strptime(month_str, "%b").month
    target_date = pd.Timestamp(date_to_view)

    # Convert Start and Lifting columns to datetime if not already
    df_pdm['Start'] = pd.to_datetime(df_pdm['Start'])
    df_pdm['Lifting'] = pd.to_datetime(df_pdm['Lifting'])

    # Apply filtering
    df_pdm_filtered = df_pdm[(df_pdm['Start'] <= target_date) & (target_date <= df_pdm['Lifting'])]
    df_full_deck_pdm_filtered = df_full_deck_pdm[(df_full_deck_pdm['Start'] <= target_date) & (target_date <= df_full_deck_pdm['Loadout_Ready'])] # this is only for until assembly area in the interior; there needs to be separate action for arriving at loadout tracks and disappearing after loadout date
    filtered_df_exterior = df_full_deck_pdm[(df_full_deck_pdm['Loadout_Ready'] <= target_date) & (target_date <= df_full_deck_pdm['Loadout'])
    ]
    df_full_deck_pdm_filtered["Zone_location"] = "Assembly_Interior"
    df_pdm_filtered["Zone_location"] = "Deck_Interior"
    filtered_df_exterior["Zone_location"] = "Assembled_Exterior"
    df = pd.concat([df_full_deck_pdm_filtered, df_pdm_filtered, filtered_df_exterior])
    df = df.sort_values(by=['Loadout_Ready', 'Lifting']).reset_index(drop=True)
    # Get a list of unique platforms
    platforms = df['Platform'].unique()
    # Create a list of colors
    colors = random.sample(list(mcolors.CSS4_COLORS.keys()), len(platforms))
    # Map each platform to a color
    platform_colors = dict(zip(platforms, colors))
    df['Color'] = df['Platform'].map(platform_colors)
    return df


def df_from_excel(date_to_view, file_path):
    xls = pd.ExcelFile(file_path)
    df_pdm = pd.read_excel(xls, sheet_name=0, header=0, parse_dates=['Start', 'Lifting', 'Loadout_Ready', 'Loadout'])
    dimensions = {
        "MAIN DECK": (22000, 38000), 
        "CELLAR DECK": (14000, 18000),
        "WELLHEAD ACCESS PLATFORM": (13000, 21000),
        "WELLHEAD DECK": (13000, 21000),
        "MEZZANINE DECK": (22000, 33000), 
        "UPPER MEZZANINE DECK": (22000, 33000), 
        "LOWER MEZZANINE DECK": (22000, 33000), 
        "HELIDECK DECK": (22000, 30000),
        "ACCESS DECK": (14000, 18000),
        "HELIDECK": (22000, 30000),
        "FULL DECK": (23000, 45000),
        "JACKET": (45000, 50000),
    }
    df_pdm.columns = ['Project', 'Platform', 'Module', 'Start', 'Lifting', 'Loadout_Ready', 'Loadout'] 
    df_full_deck_pdm = df_pdm[df_pdm['Module'] == 'FULL DECK'][['Project','Platform', 'Module', 'Start',  'Loadout_Ready','Loadout']] 
    df_pdm = df_pdm[df_pdm['Module'] != 'FULL DECK'] 
    df_pdm = df_pdm.drop(columns=['Loadout_Ready'], axis=1)  # Drop specific columns, 'Loadout'
    for df in [df_full_deck_pdm, df_pdm]:
        df['dimensions'] = df['Module'].map(dimensions)
        
    # month_num = datetime.datetime.strptime(month_str, "%b").month
    target_date = pd.Timestamp(date_to_view)
    
    # Convert Start and Lifting columns to datetime if not already
    df_pdm['Start'] = pd.to_datetime(df_pdm['Start'])
    df_pdm['Lifting'] = pd.to_datetime(df_pdm['Lifting'])
    df_full_deck_pdm['Loadout'] = pd.to_datetime(df_full_deck_pdm['Loadout'])
    df_full_deck_pdm['Loadout_Ready'] = pd.to_datetime(df_full_deck_pdm['Loadout_Ready'])
    
    # Apply filtering
    df_pdm_filtered = df_pdm[(df_pdm['Start'] <= target_date) & (target_date <= df_pdm['Lifting'])]
    df_full_deck_pdm_filtered = df_full_deck_pdm[(df_full_deck_pdm['Start'] <= target_date) & (target_date <= df_full_deck_pdm['Loadout_Ready'])] # this is only for until assembly area in the interior; there needs to be separate action for arriving at loadout tracks and disappearing after loadout date
    filtered_df_exterior = df_full_deck_pdm[(df_full_deck_pdm['Loadout_Ready'] <= target_date) & (target_date <= df_full_deck_pdm['Loadout'])]
    df_full_deck_pdm_filtered["Zone_location"] = "Assembly_Interior"
    df_pdm_filtered["Zone_location"] = "Deck_Interior"
    filtered_df_exterior["Zone_location"] = "Assembled_Exterior"
    df = pd.concat([df_full_deck_pdm_filtered, df_pdm_filtered, filtered_df_exterior])
    df = df.sort_values(by=['Loadout_Ready', 'Lifting']).reset_index(drop=True)
    
    # Get a list of unique projects
    projects = df['Project'].unique()
    # Create a list of colors
    colors = random.sample(list(mcolors.CSS4_COLORS.keys()), len(projects))
    # Map each project to a color
    project_colors = dict(zip(projects, colors))
    df['Colour'] = df['Project'].map(project_colors)

	# Fallback Loadout date: use Loadout > Loadout_Ready > Lifting
    df['Effective_Loadout'] = (
		pd.to_datetime(df['Loadout'], errors='coerce')
		.combine_first(pd.to_datetime(df['Loadout_Ready'], errors='coerce'))
		.combine_first(pd.to_datetime(df['Lifting'], errors='coerce'))
	)

	# Get the most recent Loadout per (Project, Platform)
    platform_order = (
		df.groupby(['Project', 'Platform'])['Effective_Loadout']
		.max()
		.reset_index()
		.sort_values(by='Effective_Loadout', ascending=True)
	)

	# Reorder the DataFrame
    ordered_rows = []
    for _, row in platform_order.iterrows():
	    proj, plat = row['Project'], row['Platform']
		
		# Get all rows for the project-platform
	    platform_df = df[(df['Project'] == proj) & (df['Platform'] == plat)].copy()
		
		# Put FULL DECK first
	    full_deck = platform_df[platform_df['Module'].str.upper() == 'FULL DECK']
	    others = platform_df[platform_df['Module'].str.upper() != 'FULL DECK']
		
	    platform_df_sorted = pd.concat([full_deck, others])
	    ordered_rows.append(platform_df_sorted)

	# Combine into final DataFrame
    reordered_df = pd.concat(ordered_rows, ignore_index=True)

	# Drop the helper column if not needed
    reordered_df = reordered_df.drop(columns=['Effective_Loadout'])

    return reordered_df

def create_block_data(df):

    blocks_data = []
    for index, row in df.iterrows():
        block = {
            "name": row['Platform'],
            "module": row['Module'],
            "zone": row['Placed_Zone'],
            "coords": row['coords'],
            "dims": row['dimensions'],
            "color": row['Colour'],
            "start": row['Start'],
            "proj": row['Project']
        }
        if row['Zone_location'] == "Assembly_Interior" and  row['Loadout_Ready']:
            block["lifting"] = row['Loadout_Ready']
        elif row['Zone_location'] == "Deck_Interior" and  row['Lifting']:
            block["lifting"] = row['Lifting']
        elif row['Zone_location'] == "Assembled_Exterior" and  row['Loadout']:
            block["lifting"] = row['Loadout']
        blocks_data.append(block)
    return blocks_data

def input_date():
    """
    Opens a calendar dialog, allows the user to select a date, and returns the selected date as 'MM/DD/YY'.
    """
    root = Tk()
    root.geometry("400x400")
    root.title("Select a Date")

    # Add Calendar
    cal = Calendar(root, selectmode='day', date_pattern='MM/dd/yy')

    cal.pack(pady=20)

    selected_date = {"value": None}  # Dictionary to store the selected date

    def grad_date():
        selected_date["value"] = cal.get_date()  # Store selected date
        root.destroy()  # Close the window

    # Add Button
    Button(root, text="Select Date", command=grad_date).pack(pady=20)

    root.mainloop()  # Start the event loop

    return selected_date["value"]  # Return the selected date

def get_zones(location, free_space_polygons):
    if location == "Assembled_Exterior":
        # NOTE:
        #  "C1" has skidding tracks
        # these zones are close to loadout tracks.
        zones = ["E1", "F1", "G1", "C1", "B1", "A1", "D1"] #Alternatively, create a generator that yields zones in a certain order, as one zone is filled up, move to the next zone
        crane_access = 18000
    elif location == "Paint":
        zones = ["PS1", "PS2", "PS3", "PS4", "PS", "BS1", "BS2", "BS3", "BS4"]  
        crane_access =  0  
    elif location == "Jacket":
        zones = [ "F1", "G1", "C1"]  
        crane_access =  0  
    else:
        zones = ["E2", "F2", "G2", "C2", "B2", "A2", "C3",  "D2", "B3", "B3_2", "A3", "A3_2", "FAB_MAT_STR", "F3", "D3", "E3", "E4", "A0", "A4", "D4", "D4_2", "weighbridge"]
        zones = sorted(zones, key=lambda x: free_space_polygons[x].area, reverse=True)
        if location == "Assembly_Interior":
            crane_access = 18000
        elif location == "Deck_Interior":
            crane_access = 5000
         
    return zones, crane_access


def is_within_bounds(placed_module, zone):
    """Check if the placed module is fully inside the zone."""
    return zone.contains(placed_module)

def compute_shift(placed_module, zone):
    """Find the shortest shift required to bring the module fully inside the zone."""
    if zone.contains(placed_module):
        return 0, 0  # Already inside

    shift_x, shift_y = (0, 0)
    module_coords = list(placed_module.exterior.coords)  # Get module edges
    
    for x, y in module_coords:
            point = Point(x, y)
            if not zone.contains(point):  # If this vertex is outside
                nearest_zone_point = nearest_points(point, zone)[1]
                shift_x += (nearest_zone_point.x - x)
                shift_y += (nearest_zone_point.y - y)
        
    shift_x /= len(module_coords)  # Average shift to avoid overcompensation
    shift_y /= len(module_coords)

    return shift_x, shift_y

def get_polygon_orientation(polygon):
    # Get the minimum rotated rectangle
    mrr = polygon.minimum_rotated_rectangle
    # Extract coordinates of the rectangle
    x, y = mrr.exterior.coords.xy
    # Find the longest edge of the rectangle
    edges = []
    for i in range(len(x)-1):
        edge_length = math.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)
        angle = math.atan2(y[i+1]-y[i], x[i+1]-x[i])
        edges.append((edge_length, angle))
    # Get angle of the longest edge
    longest_edge = max(edges, key=lambda x: x[0])
    return longest_edge[1]

def align_polygons(polygon1, polygon2):
    # Get orientation angles
    angle1 = get_polygon_orientation(polygon1)
    angle2 = get_polygon_orientation(polygon2)
    # Calculate angle difference in radians
    angle_diff = angle2 - angle1
    # Convert to degrees
    angle_diff_degrees = math.degrees(angle_diff)
    # Rotate polygon2 to align with polygon1
    aligned_polygon2 = rotate(polygon2, -angle_diff_degrees, origin='centroid')
    return aligned_polygon2, angle_diff_degrees


def compute_push_inside(placed_module, zone):
    """Push the module fully inside the zone if it's still overlapping."""
    intersection = placed_module.intersection(zone)
    if not intersection.is_empty and intersection.area < placed_module.area:
        # The module is partially outside
        module_coords = list(placed_module.exterior.coords)  # Get module edges
        zone_coords = list(zone.exterior.coords)  # Get zone edges
        shift_x, shift_y = 0, 0
        for x, y in module_coords:
            point = Point(x, y)
            if not zone.contains(point):  # If this vertex is outside
                nearest_zone_point = nearest_points(point, zone)[1]
                shift_x += (nearest_zone_point.x - x)
                shift_y += (nearest_zone_point.y - y)
        shift_x /= len(module_coords)  # Average shift to avoid overcompensation
        shift_y /= len(module_coords)
        #print(f"Shift needed to avoid intersection: {shift_x}, {shift_y}")
        return shift_x, shift_y

    return 0, 0  # No further push needed

def is_valid(new_polygon, placed_polygons, zone, shift=5000):
        """Check if the polygon is within bounds, has no overlaps, and satisfies crane access."""
        return (
            is_within_bounds(new_polygon, zone) and
            not any(new_polygon.intersects(p['coords']) for p in placed_polygons) and not any(
                abs(new_polygon.bounds[0] - p['coords'].bounds[2]) < shift or
                abs(new_polygon.bounds[2] - p['coords'].bounds[0]) < shift or
                abs(new_polygon.bounds[1] - p['coords'].bounds[3]) < shift or
                abs(new_polygon.bounds[3] - p['coords'].bounds[1]) < shift
                for p in placed_polygons
            )
        )

def is_valid_spacing(new_polygon, placed_polygons, zone, shift=5000):
	return (not any(
                abs(new_polygon.bounds[0] - p['coords'].bounds[2]) < shift or
                abs(new_polygon.bounds[2] - p['coords'].bounds[0]) < shift or
                abs(new_polygon.bounds[1] - p['coords'].bounds[3]) < shift or
                abs(new_polygon.bounds[3] - p['coords'].bounds[1]) < shift
                for p in placed_polygons)
)
def check_if_within(placed_module, zone):
    """Ensure the module is fully inside the zone using precise edge-based adjustments."""

    attempts = 0
    MAX_ATTEMPTS = 20  # Prevent infinite loops
    while not is_within_bounds(placed_module, zone) and attempts < MAX_ATTEMPTS:
        shift_x, shift_y = compute_shift(placed_module, zone)
        # Move polygon in a non-colliding direction
        shifts = [(shift_x + 5000, shift_y + 5000), (shift_x - 5000, shift_y - 5000), (shift_x + 5000, shift_y - 5000), (shift_x - 5000, shift_y + 5000)]
        placed_modules_try = [translate(placed_module, xoff=shifts[i][0], yoff=shifts[i][1]) for i in range(0, len(shifts))]
        # Prioritize keeping the polygon within bounds
        for new_polygon in placed_modules_try:
            if is_within_bounds(new_polygon, zone):
                new_aligned_polygon, angle_diff = align_polygons(zone, new_polygon)
                if angle_diff > 5:
                    #print(f"Polygon was rotated by {angle_diff} degrees to fit inside the zone.")
                    return new_aligned_polygon, angle_diff
                else:
                    return new_polygon, 0
        # If it's still intersecting, push it further inside using edge data
        if not is_within_bounds(placed_module, zone):
            push_x, push_y = compute_push_inside(placed_module, zone)
            placed_module = translate(placed_module, xoff=push_x, yoff=push_y)  
        attempts += 1

    #if not is_within_bounds(placed_module, zone):
        #print("Warning: Module placement failed after multiple attempts.")
        

    return placed_module, 0

def place_module_in_zone_horiz_and_verti(module, zone, target_zone, placed_modules, crane_access=5):
    min_x, min_y, max_x, max_y = zone.bounds  # Bounds of Zone
    top_left_x = min_x + 5000
    top_left_y = max_y - 5000
    l, w = module['dimensions']
    zone_area = zone.area #because zone is a polygon return from loc function
            
    module_polygon = box(0, 0, l, w)  # Create a rectangular module
    module_area = l * w
    # Check if the module can fit in the zone
    if l > (zone.bounds[2] - zone.bounds[0]) and w > (zone.bounds[3] - zone.bounds[1]):
        #print(f"Module {module} cannot fit in the zone.")
        return None
    
    #loading the initial values for each zone, namely top left, count, current row position, and current right position 
    current_row_y = zones_data[target_zone]['current_row_y']
    current_right_x = zones_data[target_zone]['current_right_x'] # this is the right boundary of the last placed module, initially it is just the top left corner of the zone
    next_top_left_x = zones_data[target_zone]['next_top_left_x']
    
    if len(placed_modules[target_zone])==0:
        placed_module = None
        current_row_y = top_left_y
        current_right_x = top_left_x + 5000
        next_top_left_x = current_right_x + crane_access
        if current_row_y - w > min_y: # checks if the module will fit in the current row heightwise
            if top_left_x + l < max_x: # checks if the module will fitin the current row lengthwise
                placed_module = translate(module_polygon, top_left_x, current_row_y - w) # place lengthwise
                placed_module, angle = check_if_within(placed_module, zone) 
                if placed_module:
                    if zone.contains(placed_module):
                        current_right_x = placed_module.bounds[2] # this is the right boundary of the last placed module
                        next_top_left_x = current_right_x + crane_access
                        print(f"First placement at top_left {placed_module.bounds[0]} in {target_zone}. Length is {l}. so current right is {current_right_x} and next will be placed at {next_top_left_x}") 
                    else:    
                        print("Polygon couldn't be placed inside the zone, trying a new empty zone.")
                        return None
                else:
                    # Calculate the position of the next module (to the right of the last placed module)
                    current_right_x = top_left_x
                    next_top_left_x = current_right_x + crane_access
                       
            else:
                #rotate and place
                current_right_x = top_left_x + 5000
                next_top_left_x = current_right_x + crane_access
                module_polygon = box(0, 0, w, l)
                placed_module = translate(module_polygon, top_left_x, current_row_y - l)
                placed_module, angle = check_if_within(placed_module, zone)
                if placed_module:
                    if zone.contains(placed_module):
                        current_right_x = placed_module.bounds[2] # this is the right boundary of the last placed module
                        next_top_left_x = current_right_x + crane_access
                        print(f"First placement at top_left {placed_module.bounds[0]} in {target_zone}. Length is {w}. so current right is {current_right_x} and next will be placed at {next_top_left_x}")        
                    else:    
                        print("Polygon couldn't be placed inside the zone, trying a new empty zone.")
                        return None
                else:
                    current_right_x = top_left_x
                    next_top_left_x = current_right_x + crane_access
                    

        # Update the variables in the dictionary after the first placement
        update_positions_in_dict(target_zone, current_row_y, current_right_x, next_top_left_x)
        return placed_module
                
    # AFTER THE FIRST PLACEMENT            
    # Check if the module fits in the current row (horizontally)
    else:   
        if current_row_y - w > min_y:
            if next_top_left_x + l < max_x:
            #Place lengthwise as first attempt
                placed_module = translate(module_polygon, next_top_left_x, current_row_y - w)
                placed_module, angle = check_if_within(placed_module, zone)
                if not is_valid(placed_module, placed_modules[target_zone], zone):
                    placed_module = resolve_overlap(placed_module, placed_modules[target_zone], zone, crane_access, module['Module'])
                
                if placed_module:
                    if not is_valid_spacing(placed_module, placed_modules[target_zone], zone):
                        placed_module = resolve_spacing(placed_module, placed_modules[target_zone], zone, crane_access, module['Module'])
                    if zone.contains(placed_module):
                        current_right_x = placed_module.bounds[2] # this is the right boundary of the last placed module
                        next_top_left_x = current_right_x + crane_access
                        print(f"next placed at {placed_module.bounds[0]} in {target_zone}. length is {l} so current right is {current_right_x}. next will be at {next_top_left_x}")
                    else:    
                        print("Polygon couldn't be placed inside the zone, trying a new empty zone.")
                        return None
                else:
                    # Update the right boundary for the next placement
                    current_right_x = next_top_left_x
                    next_top_left_x = current_right_x + crane_access
                    
                # Update the variables in the dictionary as well
                update_positions_in_dict(target_zone, current_row_y, current_right_x, next_top_left_x)
                return placed_module
            
            else:
                #next row
                current_row_y -= w
                next_top_left_x = min_x   # Start from the left again for the new row
                if current_row_y - w > min_y:
                    if next_top_left_x + l < max_x:
                        placed_module = translate(module_polygon, next_top_left_x, current_row_y - w)
                        placed_module, angle = check_if_within(placed_module, zone)
                        if not is_valid(placed_module, placed_modules[target_zone], zone):
                            placed_module = resolve_overlap(placed_module, placed_modules[target_zone], zone, crane_access, module['Module'])

                        if placed_module:
                            if not is_valid_spacing(placed_module, placed_modules[target_zone], zone):
                                placed_module = resolve_spacing(placed_module, placed_modules[target_zone], zone, crane_access, module['Module'])
                            if zone.contains(placed_module):
                                current_right_x = placed_module.bounds[2] # this is the right boundary of the last placed module
                                next_top_left_x = current_right_x + crane_access
                                
                                print(f"Module placed in next row at {placed_module.bounds[0]} in {target_zone}. Length is {l} and vertically at {current_row_y}. Next will be at {next_top_left_x}")
                            else:    
                                print("Polygon couldn't be placed inside the zone, trying a new empty zone.")
                                return None
                        else:
                            current_right_x = next_top_left_x  # Reset for the new row
                            next_top_left_x = current_right_x + crane_access
                            
                        # Update the variables in the dictionary as well
                        update_positions_in_dict(target_zone, current_row_y, current_right_x, next_top_left_x)
                        
                        return placed_module
                    
        elif current_row_y - l > min_y:
            if next_top_left_x + w < max_x:
                #rotate and place
                module_polygon = box(0, 0, w, l)
                placed_module = translate(module_polygon, next_top_left_x, current_row_y - l)
                placed_module, angle = check_if_within(placed_module, zone)
                if not is_valid(placed_module, placed_modules[target_zone], zone):
                    placed_module = resolve_overlap(placed_module, placed_modules[target_zone], zone, crane_access, module['Module'])

                if placed_module:
                    if not is_valid_spacing(placed_module, placed_modules[target_zone], zone):
                        placed_module = resolve_spacing(placed_module, placed_modules[target_zone], zone, crane_access, module['Module'])
                    if zone.contains(placed_module):
                        current_right_x = placed_module.bounds[2] # this is the right boundary of the last placed module
                        next_top_left_x = current_right_x + crane_access
                        print(f"Module placed at {placed_module.bounds[0]} in {target_zone}. Length is {w}. so current right is {current_right_x}. Next will be at {next_top_left_x}")
                    else:    
                        print("Polygon couldn't be placed inside the zone, trying a new empty zone.")
                        return None
                else:    
                    current_right_x = next_top_left_x
                    next_top_left_x = current_right_x + crane_access
                
                # Update the variables in the dictionary as well
                update_positions_in_dict(target_zone, current_row_y, current_right_x, next_top_left_x)
                return placed_module
        
            # Check if there's enough vertical space in the zone
        else:
            print(f"Module does not fit in Zone {target_zone}, trying next")
            return None


def is_within_bounds(polygon, des_zone):
    """Check if a polygon is fully within the main polygon."""
    return des_zone.contains(polygon)

def resolve_overlap(polygon, placed_polygons, des_zone, crane_access, module_type):
    """Try shifting the polygon to resolve overlap while staying inside the main polygon."""
    attempts = 20
    while attempts > 0:
        new_polygon = resolve_overlap_once(polygon, placed_polygons, des_zone, crane_access, module_type)
        if new_polygon is not None:
            print("Overlap resolved")
            return new_polygon
        attempts -= 1
    print("Couldn't resolve overlap")
    return None

def resolve_overlap_once(polygon, placed_polygons, zone, crane_access, module_type):
    """
    Resolves overlap by trying to shift both the polygon being placed and the overlapping polygon.
    
    Args:
        polygon: The polygon being placed.
        placed_polygons: List of existing placed polygons, each a dict with 'coords' and 'type'.
        main_polygon: The boundary polygon.
        crane_access: Minimum horizontal distance required between polygons.
        module_type: Type of the module for comparison.
    
    Returns:
        The updated polygon after resolving overlap, or None if no valid position is found.
    """
    
    #if not is_valid(polygon, placed_polygons):
        #return None 

    for other_polygon in placed_polygons:
        if polygon.intersects(other_polygon['coords']):
            # Calculate intersection and required shift
            intersection = polygon.intersection(other_polygon['coords'])
            dx = abs(intersection.bounds[2] - intersection.bounds[0])  # Overlap width
            dy = abs(intersection.bounds[3] - intersection.bounds[1])  # Overlap height
            # Apply crane access or large shift based on module type
            shift = crane_access if module_type == other_polygon['type'] else 5000
            print(f"Overlap of {module_type} detected with {other_polygon['type']}. Shifting by {dx+shift}")
            dx += shift
            #dy += shift

            # Create movement options
            shifts = [
                (dx, 0), (0, -dy), (0, dy), (-dx, 0),
                (dx, dy), (-dx, -dy), (dx, -dy), (-dx, dy)
            ]
            rotations = [90, 180, 270, 360]   
            
			# Try shifting the current polygon
            for xoff, yoff in shifts:
                new_polygon = translate(polygon, xoff=xoff, yoff=yoff)
                if is_valid(new_polygon, placed_polygons, zone):
                    
                    return new_polygon
                else:
                    for rotation in rotations:
                        new_polygon_rot = rotate(polygon, -rotation, origin='centroid')
                        if is_valid(new_polygon_rot, placed_polygons, zone):
                            
                            return new_polygon_rot
	            
            print("Warning: Could not resolve overlap cleanly")
            return None  # No valid move found
	    
    # No overlap found, return the original polygon
    return polygon

def resolve_spacing(polygon, placed_polygons, des_zone, crane_access, module_type):
    """Try shifting the polygon to resolve overlap while staying inside the main polygon."""
    attempts = 10
    while attempts > 0:
        new_polygon = resolve_spacing_with_buffer(polygon, placed_polygons, des_zone)
        #new_polygon = resolve_spacing_once(polygon, placed_polygons, des_zone)
        if new_polygon is not None:
            print("Conditions are valid; spacing problem resolved.")
            return new_polygon
        attempts -= 1
    
    print("Couldn't resolve spacing")
    return None


def resolve_spacing_with_buffer(polygon, placed_polygons, zone,
                                min_x_spacing=8000, min_y_spacing=1000,
                                max_attempts=10, shift_step=1000):
    """
    Ensures buffer-based spacing using shapely's buffer() and scale().
    """

    shift_x = 0
    shift_y = 0


        # Create a custom "buffer" region for spacing using scaling
    spacing_zone = scale(polygon, 
                             xfact=(polygon.bounds[2] - polygon.bounds[0] + min_x_spacing) / (polygon.bounds[2] - polygon.bounds[0]),
                             yfact=(polygon.bounds[3] - polygon.bounds[1] + min_y_spacing) / (polygon.bounds[3] - polygon.bounds[1]),
                             origin='center')

    spacing_ok = True
    for placed in placed_polygons:
        if spacing_zone.intersects(placed['coords']):
            spacing_ok = False
            break

    if spacing_ok:
        return polygon

    # Try shifting the polygon to resolve spacing
    shift_x += shift_step
    shift_y += shift_step
    polygon = translate(polygon, xoff=shift_x, yoff=shift_y)


    return polygon


def resolve_spacing_once(polygon, placed_polygons, zone):
    for other_polygon in placed_polygons:
        # Calculate intersection and required shift
        #dx = max(abs(polygon.bounds[0] - other_polygon['coords'].bounds[2]), abs(polygon.bounds[2] - other_polygon['coords'].bounds[0]))
        #dy = min(abs(polygon.bounds[1] - other_polygon['coords'].bounds[3]), abs(polygon.bounds[3] - other_polygon['coords'].bounds[1]))
        dx = 1000
        dy = 1000
        # Create movement options
        shifts = [(dx, 0), (0, -dy), (0, dy), (-dx, 0),
            (dx, dy), (-dx, -dy), (dx, -dy), (-dx, dy)]
        rotations = [90, 180, 270, 360]   
        
		# Try shifting the current polygon
        for xoff, yoff in shifts:
            new_polygon = translate(polygon, xoff=xoff, yoff=yoff)
            if is_valid_spacing(new_polygon, placed_polygons, zone):
                
                return new_polygon
            else:
                for rotation in rotations:
                    new_polygon_rot = rotate(polygon, -rotation, origin='centroid')
                    if is_valid_spacing(new_polygon_rot, placed_polygons, zone):
                        
                        return new_polygon_rot
        

        print("Warning: Could not resolve spacing cleanly")
        return None  # No valid move found
	    
    # No overlap found, return the original polygon
    return polygon

def update_positions_in_dict(target_zone, current_row_y, current_right_x, next_top_left_x):
    zones_data[target_zone]['current_row_y'] = current_row_y
    zones_data[target_zone]['current_right_x'] = current_right_x
    zones_data[target_zone]['next_top_left_x'] = next_top_left_x

def iterate_over_zones_and_place(zones, df, row, free_space_polygons, placed_modules, crane_access, idx):
    flag = 0
    #preferred_zones = sorted(zones, key=lambda z: 0 if any(row['Project'] == p.get('Project') for p in placed_modules[z]) else 1)
    
    preferred_zones = sorted(zones, key=lambda z: -sum(1 for p in placed_modules[z] if p.get('Project') == row['Project']))

    print(row, preferred_zones)
    for zone in preferred_zones:
        module_coords = place_module_in_zone_horiz_and_verti(row, free_space_polygons[zone], zone, placed_modules, crane_access=crane_access)
        if module_coords:
            df.at[idx, 'coords'] = module_coords
            df.at[idx, 'Placed_Zone'] = zone
            placed_modules[zone].append({"type" : row['Module'], "idx": idx, "coords" : module_coords, "Project": row['Project'] })
            flag = 1
            break  # Successfully placed, move to the next module
            
        else:
            continue
    return flag

def get_coords_of_each_module(df, free_space_polygons, placed_modules, not_placed, zones_data):
    #for proj in sorted(set(r["Project"] for idx, r in rects)):
        #proj_rects = [(idx, r) for idx, r in rects if r["Project"] == proj]
        #print(proj_rects)
    for idx, row in df.iterrows():
        zones, crane_access = get_zones(row['Zone_location'], free_space_polygons)
        print(f"Processing: {row['Platform']} {row['Module']} -> Potential Zones: {zones}")

        flag = iterate_over_zones_and_place(zones, df, row, free_space_polygons, placed_modules, crane_access, idx)
        if flag ==1:
            continue
        elif flag == 0:
            print(f"Module {row['Platform']} {row['Module']} couldn't be placed in any zone.")
            if row['Zone_location'] == "Paint":
                other_zone = 'Deck_Interior'
                zones, crane_access = get_zones(other_zone, free_space_polygons)
                flag = iterate_over_zones_and_place(zones, df, row, free_space_polygons, placed_modules, crane_access, idx)
                if flag == 0:
                    print(f"Module {row['Platform']} {row['Module']} couldn't be placed in any zone.")
                    not_placed = pd.concat([not_placed, pd.DataFrame([row])], ignore_index=True)
            elif row['Zone_location'] == "Assembly_Interior":
                print("Couldn't be placed inside, so placing in outer zones")
                other_zone = 'Assembled_Exterior'
                zones, crane_access = get_zones(other_zone, free_space_polygons)
                flag = iterate_over_zones_and_place(zones, df, row, free_space_polygons, placed_modules, crane_access, idx)
                if flag == 0:
                    print(f"Module {row['Platform']} {row['Module']} couldn't be placed in any zone.")
                    not_placed = pd.concat([not_placed, pd.DataFrame([row])], ignore_index=True)
            else:
                not_placed = pd.concat([not_placed, pd.DataFrame([row])], ignore_index=True)

    return df, placed_modules, not_placed



def space_utilization(zones_data, free_space_polygons, placed_modules):
    space_util_rows = []

    # Smallest module dimensions + margin
    min_length = 14000
    min_width = 18000

    for zone, data in zones_data.items():
        zone_polygon = free_space_polygons[zone]
        zone_area = zone_polygon.area

        # Calculate actual occupied area
        occupied_area = sum(mod['coords'].area for mod in placed_modules.get(zone, []))
        free_area = zone_area - occupied_area

        # Get zone bounds
        zone_min_x, zone_min_y, zone_max_x, zone_max_y = zone_polygon.bounds

        utilization = (occupied_area / zone_area) * 100 if zone_area > 0 else 0
        

        space_util_rows.append({
            'Zone': zone,
            'Area': round(zone_area/1000000, 2),
            'Occupied_Area': round(occupied_area/1000000, 2),
            'Free_Area': round(free_area/1000000, 2),
            'Utilization': round(utilization, 2),
            'Module_Count': len(placed_modules[zone])
        })

    space_util_report = pd.DataFrame(space_util_rows)

    return space_util_report

def show_utilization_report(df):
	# Custom headers with symbols
    header_mapping = {
        'Zone': 'Zone',
        'Area': 'Area (m²)',
        'Occupied_Area': 'Occupied Area (m²)',
        'Free_Area': 'Free Area (m²)',
        'Utilization': 'Utilization (%)',
        'Module_Count': 'Module Count'
    }
    # Create a table widget and populate it
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Space Utilization Report")
    layout = QtWidgets.QVBoxLayout()

    table = QtWidgets.QTableWidget()
    table.setRowCount(len(df))
    table.setColumnCount(len(df.columns))
    display_headers = [header_mapping.get(col, col) for col in df.columns]
    table.setHorizontalHeaderLabels(display_headers)
    table.horizontalHeader().setStretchLastSection(True)

    for row in range(len(df)):
        for col, column_name in enumerate(df.columns):
            item = QtWidgets.QTableWidgetItem(str(df.iloc[row][column_name]))
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            table.setItem(row, col, item)

    layout.addWidget(table)
    dialog.setLayout(layout)
    dialog.resize(700, 400)
    dialog.exec_() 

def cleanup_macro_created():
    """Remove all objects tagged by the macro in the previous run."""
    doc = FreeCAD.ActiveDocument
    if not doc:
        print("No active document.")
        return
    
    to_delete = [obj for obj in doc.Objects if hasattr(obj, "macroCreated") and obj.macroCreated]
    
    for obj in to_delete:
        doc.removeObject(obj.Name)
        print(f"Removed previous macro-created object")


doc = FreeCAD.ActiveDocument  # Ensure your layout is open
cleanup_macro_created()
date_to_view = input_date()

zones_data = {}

rows = list(string.ascii_uppercase[:8])  # A to H
cols = range(1, 5)  # 1 to 4

for row in rows:
    for col in cols:
        zone_key = f"{row}{col}"
        zones_data[zone_key] = {
            'current_row_y': 0,
            'current_right_x': 0,
            'next_top_left_x': 0
        }
# remove ('B4', 'C4', 'F4', 'G3', 'G4', 'H4', 'H3', 'H2') from zones_data.
zones_data = {k: v for k, v in zones_data.items() if k not in ('B4', 'C4', 'F4', 'G3', 'G4', 'H4', 'H3', 'H2')}
for zon in ['A0', 'FAB_MAT_STR', 'BS1', 'BS2', 'BS3', 'BS4', 'PS_1', 'PS_2', 'PS_3', 'PS_4', 'weighbridge', 'A3_2', 'B3_2', 'D4_2']:
    zones_data[zon] = {'current_row_y': 0, 'current_right_x': 0, 'next_top_left_x': 0}

# Step 1: Create Blocks to represent the modules 
file = 'C:/Users/amrin.kareem/Downloads/work/Yard_Optimization/OneDrive_2024-10-16/P_2300_2302_PDM_only.xlsx'
file_path = 'C:/Users/amrin.kareem/Downloads/work/Yard_Optimization/df_pdm_deck.xlsx'
df = df_from_excel(date_to_view, file_path)

# Step 2: Identify Free Spaces
target_polygon_names = ["A1", "A0", "A2", "A3", "A3_2", "B1", "B2", "B3", "B3_2", "C1", "C2", "C3", "D1", "D2", "D3", "D4", "D4_2", "E1", "E2", "E3", "E4", "F1", "F2", "F3", "G1", "G2", "A4", "H1", "weighbridge", "PS_1", "PS_2", "PS_3", "PS_4", "BS1", "BS2", "BS3", "BS4", "FAB_MAT_STR"]
free_space_polygons = compute_free_spaces_from_poly_names(target_polygon_names)
placed_modules = {zone: [] for zone in zones_data}  # Dynamically initialize zones
not_placed = pd.DataFrame()
df, placed_modules, not_placed = get_coords_of_each_module(df, free_space_polygons, placed_modules, not_placed, zones_data)
space_util_report = space_utilization(zones_data, free_space_polygons, placed_modules)
blocks_data = create_block_data(df)

# Step 3: Place modules from DataFrame onto free spaces 
place_blocks_in_free_space(blocks_data) # Place blocks in the free space
create_gantt_chart(blocks_data)
# Show the report in a separate Qt window
show_utilization_report(space_util_report)
doc.recompute()  # Update the FreeCAD document