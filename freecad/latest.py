from shapely.affinity import translate
from shapely.geometry import box, Point
from shapely.ops import nearest_points
import FreeCAD as App
import re
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
import pandas as pd
import Draft  # For draft tools like adding blocks
from shapely.geometry import Polygon, Point, box
from shapely.affinity import translate, rotate
import string
font_path = r"C:\Users\amrin.kareem\Downloads\work\NMDC-Team-Work\YardLayout\consolas\consola.ttf"

def create_project_block(project_name, module, polygon, color = [255, 255, 0]):
    """
    Creates a block (polygon) in FreeCAD with the given name, coordinates, and color.

    :param project_name: Name of the project (used as the block's name in FreeCAD)
    :param polygon_coords: List of tuples representing the polygon vertices [(x1, y1), (x2, y2), ...]
    :param color_code: List of three integers representing the RGB color code of the block.
    :return: Part::Feature object of the created block.
    """
    doc = App.ActiveDocument
    print(f"Creating block for {project_name} {module} with coords {polygon} and color {color}")
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
    block = doc.addObject("Part::Feature", project_name+"\n"+module)
    block.Shape = polygon_face
		
	# Compute centroid of the polygon (Shapely)
    centroid = polygon.centroid
    centroid_vector = App.Vector(centroid.x, centroid.y, 0)

    # Create text at the centroid
    text_obj = Draft.make_shapestring(f"{project_name} {module}", font_path, 2000, 5)

	# Offset text vertically (stack one above the other)
    text_spacing = 2000  # Spacing between lines
    min_x = min(p.x for p in polygon_vectors)  # Left edge
    centroid_y = polygon.centroid.y  # Middle Y position

    # Create first line of text (project_name)
    text1 = Draft.make_shapestring(project_name, font_path, 2000, 5)
    text1.Placement.Base = App.Vector(min_x + 1000, centroid_y + text_spacing, 0)  # Shift up

    # Create second line of text (module)
    text2 = Draft.make_shapestring(module, font_path, 2000, 5)
    text2.Placement.Base = App.Vector(min_x + 1000, centroid_y - text_spacing, 0)  # Shift down
    
	# Set block color
    block.ViewObject.ShapeColor = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    block.ViewObject.DiffuseColor = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
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
        dimensions = block_data["dims"]
        color = block_data["color"]
        
        # Create a block (rectangle)
        create_project_block(project_name, module, coords)
        

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
    """
    Function to read Excel file and generate a dictionary of platforms
    (keys) and their corresponding module data (values).

    Parameters
    ----------
    date_to_view : str
        Date to view in 'DD MMM YYYY' format (e.g., 11 Sep 2024).
    file : str
        Path to the Excel file.

    Returns
    -------
    df_pdm_filtered : pandas.DataFrame
        Filtered DataFrame containing platforms and their modules that are
        under construction during the target date.
    df_full_deck_pdm_filtered : pandas.DataFrame
        Filtered DataFrame containing platforms and their FULL DECK modules
        that are under construction during the target date.
    filtered_df_exterior : pandas.DataFrame
        Filtered DataFrame containing platforms and their FULL DECK modules
        that are at the exterior of the yard during the target date.

    Notes
    -----
    The Excel file 'PDM' sheet should contain the data for the platforms.

    The data should have the following columns:

    * 'Project'
    * 'Title'
    * 'SubModule'
    * 'Main Module'
    * 'Starting Assembly'
    * 'Lifting_Date'
    * 'Loadout_Date'
    * 'Seafastening'

    """
    
    xls = pd.ExcelFile(file)
    df_pdm = pd.read_excel(xls, header=1, parse_dates=['Pancake Assembly Start Date', 'Pancake Stacking date', 'Pancake Painting Start date', 'Pancake Painting End date', 'Stacking Preparation date', 'PDM Loadout Ready Date', 'PDM Load Out date'])
    dimensions = {
        "MAIN DECK": (22000, 38000), 
        "CELLAR DECK": (18000, 14000),
        "WELLHEAD ACCESS PLATFORM": (13000, 21000),
        "MEZZANINE DECK": (33000, 22000), 
        "HELIDECK DECK": (30000, 22000),
        "FULL DECK": (23000, 45000)
    }
    df_pdm.columns = ['Project', 'Platform', 'Module', 'Start', 'Lifting', 'Paint_Start', 'Paint_End', 'Loadout_Ready', 'Loadout']
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
    df_pdm['Paint_Start'] = pd.to_datetime(df_pdm['Paint_Start'])
    df_pdm['Paint_End'] = pd.to_datetime(df_pdm['Paint_End'])
    df_pdm['Loadout'] = pd.to_datetime(df_pdm['Loadout'])
    df_pdm['Loadout_Ready'] = pd.to_datetime(df_pdm['Loadout_Ready'])

    # Apply filtering
    df_pdm_filtered = df_pdm[(df_pdm['Start'] <= target_date) & (target_date <= df_pdm['Lifting']) & (df_pdm['Paint_Start'] >= df_pdm['Lifting'])] # should be taken to assembly area and painted at loaction
    df_full_deck_pdm_filtered = df_full_deck_pdm[(df_full_deck_pdm['Start'] <= target_date) & (target_date <= df_full_deck_pdm['Loadout_Ready'])] # this is only for until assembly area in the interior; there needs to be separate action for arriving at loadout tracks and disappearing after loadout date
    df_to_paint = df_pdm[(df_pdm['Paint_Start'] <= target_date) & (target_date <= df_pdm['Paint_End']) & ((df_pdm['Paint_End'] <= df_pdm['Lifting']))] # should be taken to painting place
    filtered_df_exterior = df_full_deck_pdm[(df_full_deck_pdm['Loadout_Ready'] <= target_date) & (target_date <= df_full_deck_pdm['Loadout'])]
        
    df_full_deck_pdm_filtered["Zone_location"] = "Assembly_Interior"
    df_pdm_filtered["Zone_location"] = "Deck_Interior"
    df_to_paint["Zone_location"] = "Paint"
    # filtered_df_exterior["Zone_location"] = "Assembled_Exterior"
    df = pd.concat([df_full_deck_pdm_filtered, df_pdm_filtered, df_to_paint, filtered_df_exterior]) 
    df = df.sort_values(by=['Loadout_Ready', 'Lifting', 'Paint_Start']).reset_index(drop=True)
    # Get a list of unique platforms
    platforms = df['Project'].unique()
    # Create a list of colors
    colors = random.sample(list(mcolors.CSS4_COLORS.keys()), len(platforms))
    # Map each platform to a color
    platform_colors = dict(zip(platforms, colors))
    df['Color'] = df['Project'].map(platform_colors)
    return df

def create_block_data(df):

    """blocks_data = [
        {"name": "Project A", "module": "FULL DECK", "zone": "A1", coords": (10, 20), "dims": (100, 150), "color": (255, 0, 0)},  # Red
        {"name": "Project B",  "module": "FULL DECK", "zone": "A2", "coords": (150, 300), "dims": (120, 200), "color": (0, 255, 0)},  # Green
        {"name": "Project C",  "module": "FULL DECK", "zone": "B3", "coords": (400, 50), "dims": (120, 180), "color": (0, 0, 255)}   # Blue
    ]
    """
    blocks_data = []
    for index, row in df.iterrows():
        block = {
            "name": row['Platform'],
            "module": row['Module'],
            "zone": row['Placed_Zone'],
            "coords": row['coords'],
            "dims": row['dimensions'],
            "color": row['Color']
        }
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

def get_zones(location):
    if location == "Assembled_Exterior":
        # NOTE:
        #  "C1" has skidding tracks
        # these zones are close to loadout tracks.
        zones = ["C1", "B1", "A1", "D1", "E1", "F1", "G1"] #Alternatively, create a generator that yields zones in a certain order, as one zone is filled up, move to the next zone
        crane_access = 18000
    elif location == "Paint":
        zones = ["PS1", "PS2", "PS3", "PS4", "PS"]  
        crane_access =  0  
    else:
        zones = ["C2", "B2", "A2", "C3", "B3", "A3", "D2", "E2", "F2", "G2", "F3", "D3", "E3", "E4", "A0", "A4", "D4"]
        if location == "Assembly_Interior":
            crane_access = 18000
        elif location == "Deck_Interior":
            crane_access = 5000
         
    return zones, crane_access

def compute_push_inside(placed_module, zone, target_point):
    """Push the top-left point of the module toward a specified target point in the zone, ensuring it's inside."""
    
    # Get the intersection between the placed module and the zone
    intersection = placed_module.intersection(zone)

    if not intersection.is_empty and intersection.area < placed_module.area:
        # The module is partially outside
        module_coords = list(placed_module.exterior.coords)  # Get module edges
        
        # Get the top-left corner (minimum x, maximum y)
        top_left_point = Point(min(module_coords, key=lambda x: x[0])[0],  # min(x) for the leftmost point
                               max(module_coords, key=lambda x: x[1])[1])  # max(y) for the topmost point
        
        # Calculate how much to shift the top-left point towards the target point
        shift_x = target_point.x - top_left_point.x
        shift_y = target_point.y - top_left_point.y
        
        # Apply the shift to the entire module
        placed_module = translate(placed_module, xoff=shift_x, yoff=shift_y)
        
        # Ensure the module is inside the zone
        if zone.contains(placed_module):
            return placed_module
        else:
            # If after shifting, it's still not inside, try again
            return compute_push_inside(placed_module, zone, target_point)

    return placed_module

def is_within_bounds(placed_module, zone):
    """Check if the placed module is fully inside the zone."""
    return zone.contains(placed_module)

def compute_shift(placed_module, zone):
    """Find the shortest shift required to bring the module fully inside the zone."""
    if zone.contains(placed_module):
        return 0, 0  # Already inside

    module_center = placed_module.centroid
    nearest_valid_point = nearest_points(module_center, zone)[1]

    shift_x = nearest_valid_point.x - module_center.x
    shift_y = nearest_valid_point.y - module_center.y

    return shift_x, shift_y

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
        
        # shift_x /= len(module_coords)  # Average shift to avoid overcompensation
        # shift_y /= len(module_coords)
        return shift_x, shift_y

    return 0, 0  # No further push needed

def check_if_within(placed_module, zone):
    """Ensure the module is fully inside the zone using precise edge-based adjustments."""
    if placed_module.area > zone.area:
        print("Error: Module is larger than the zone and cannot fit.")
        return None  # No possible placement

    attempts = 0
    MAX_ATTEMPTS = 20  # Prevent infinite loops
    while not is_within_bounds(placed_module, zone) and attempts < MAX_ATTEMPTS:
        shift_x, shift_y = compute_shift(placed_module, zone)
        placed_module = translate(placed_module, xoff=shift_x, yoff=shift_y)

        # If it's still intersecting, push it further inside using edge data
        if (shift_x, shift_y) == (0,0) and not is_within_bounds(placed_module, zone):
            push_x, push_y = compute_push_inside(placed_module, zone)
            placed_module = translate(placed_module, xoff=push_x, yoff=push_y)
        attempts += 1

    if not is_within_bounds(placed_module, zone):
        print("Warning: Module placement failed after multiple attempts.")
        

    return placed_module

def place_module_in_zone_horiz_and_verti(module, zone, target_zone, placed_modules, crane_access=5):
    
    min_x, min_y, max_x, max_y = zone.bounds  # Bounds of Zone
    top_left_x = min_x + 5000
    top_left_y = max_y - 5000
    l, w = module['dimensions']
    zone_area = zone.area #because zone is a polygon return from loc function
            
    module_polygon = box(0, 0, l, w)  # Create a rectangular module
    module_area = l * w
    # Check if the module can fit in the zone
    if module_area > zone_area:
        print(f"Module {module} cannot fit in the zone.")
        return None
    
    #loading the initial values for each zone, namely top left, count, current row position, and current right position 
    count = zones_data[target_zone]['count']
    current_row_y = zones_data[target_zone]['current_row_y']
    current_right_x = zones_data[target_zone]['current_right_x'] # this is the right boundary of the last placed module, initially it is just the top left corner of the zone
    next_top_left_x = zones_data[target_zone]['next_top_left_x']
    
    if count==0:
        placed_module = None
        current_row_y = top_left_y
        current_right_x = top_left_x + l
        next_top_left_x = current_right_x + crane_access
        if current_row_y - w > min_y: # checks if the module will fit in the current row heightwise
            if top_left_x + l < max_x: # checks if the module will fitin the current row lengthwise
                placed_module = translate(module_polygon, top_left_x, current_row_y - w) # place lengthwise
                placed_module = check_if_within(placed_module, zone) 
                if placed_module:
                    if zone.contains(placed_module):
                        count += 1
                    else:
                        print("Polygon couldn't be placed inside the zone, trying a new empty zone.")
                        return None
                # Calculate the position of the next module (to the right of the last placed module)
                current_right_x = top_left_x + l
                next_top_left_x = current_right_x + crane_access
                print(f"First placement at top_left {top_left_x} in {target_zone}. Length is {l}. so current right is {current_right_x} and next will be placed at {next_top_left_x}")        
            else:
                #rotate and place
                current_right_x = top_left_x + w
                next_top_left_x = current_right_x + crane_access
                module_polygon = box(0, 0, w, l)
                placed_module = translate(module_polygon, top_left_x, current_row_y - l)
                placed_module = check_if_within(placed_module, zone)
                if placed_module:
                    if zone.contains(placed_module):
                        count += 1
                    else:
                        print("Polygon couldn't be placed inside the zone, trying a new empty zone.")
                        return None
                current_right_x = top_left_x + w
                next_top_left_x = current_right_x + crane_access
                print(f"First placement at top_left {top_left_x} in {target_zone}. Length is {w}. so current right is {current_right_x} and next will be placed at {next_top_left_x}")        

        # Update the variables in the dictionary after the first placement
        update_positions_in_dict(target_zone, count, current_row_y, current_right_x, next_top_left_x)
        return placed_module
                
    # AFTER THE FIRST PLACEMENT            
    # Check if the module fits in the current row (horizontally)
    else:   
        if current_row_y - w > min_y:
            if next_top_left_x + l < max_x:
            #Place lengthwise as first attempt
                placed_module = translate(module_polygon, next_top_left_x, current_row_y - w)
                placed_module = check_if_within(placed_module, zone)
                
                print(f"next placed at {next_top_left_x} in {target_zone}. length is {l} so current right is {next_top_left_x + l}. next will be at {next_top_left_x + l + crane_access}")
                #CHECK IF THERE IS OVERLAP WITH ANY OF THE PREVIOUSLY PLACED MODULES
                # Resolve overlap iteratively
                placed_module = resolve_overlap(placed_module, placed_modules[target_zone], zone)
                if placed_module:
                    # If it's still valid, add to placed polygons
                    if zone.contains(placed_module):
                        count += 1
                    else:
                        print("Polygon couldn't be placed properly, trying a new empty zone.")
                        return None
                
                # Update the right boundary for the next placement
                current_right_x = next_top_left_x + l
                next_top_left_x = current_right_x + crane_access

                # Update the variables in the dictionary as well
                update_positions_in_dict(target_zone, count, current_row_y, current_right_x, next_top_left_x)
                return placed_module
            
            else:
                #next row
                current_row_y -= (w + 5000)
                next_top_left_x = min_x + 5000  # Start from the left again for the new row
                if current_row_y - w > min_y:
                    if next_top_left_x + l < max_x:
                        placed_module = translate(module_polygon, next_top_left_x, current_row_y - w)
                        placed_module = check_if_within(placed_module, zone)
                        
                        print(f"Module placed in next row at {next_top_left_x} in {target_zone}. Length is {l} and vertically at {current_row_y}. Next will be at {next_top_left_x + l + crane_access}")
                        placed_module = resolve_overlap(placed_module, placed_modules[target_zone], zone)
                        if placed_module:
                            # If it's still valid, add to placed polygons
                            if zone.contains(placed_module):
    
                                count += 1
                            else:
                                print("Polygon couldn't be placed properly, trying a new empty zone.")
                                return None
                
                        current_right_x = next_top_left_x + l  # Reset for the new row
                        next_top_left_x = current_right_x + crane_access
                        count += 1
                        # Update the variables in the dictionary as well
                        update_positions_in_dict(target_zone, count, current_row_y, current_right_x, next_top_left_x)
                        
                        return placed_module
                    
        elif current_row_y - l > min_y:
            if next_top_left_x + w < max_x:
                #rotate and place
                module_polygon = box(0, 0, w, l)
                placed_module = translate(module_polygon, next_top_left_x, current_row_y - l)
                placed_module = check_if_within(placed_module, zone)
                print(f"Module placed at {next_top_left_x} in {target_zone}. Length is {w}. so current right is {next_top_left_x + w}. Next will be at {next_top_left_x + w + crane_access}")
                placed_module = resolve_overlap(placed_module, placed_modules[target_zone], zone)
                if placed_module:
                    # If it's still valid, add to placed polygons
                    if zone.contains(placed_module):
                        count += 1
                    else:
                        print("Polygon couldn't be placed properly, trying a new empty zone.")
                        return None
                current_right_x = next_top_left_x + w
                next_top_left_x = current_right_x + crane_access
                # Update the variables in the dictionary as well
                update_positions_in_dict(target_zone, count, current_row_y, current_right_x, next_top_left_x)
                return placed_module
        
            # Check if there's enough vertical space in the zone
        else:
            print(f"Module does not fit in Zone {target_zone}, trying next")
            return None


def is_within_bounds(polygon, main_polygon):
    """Check if a polygon is fully within the main polygon."""
    return main_polygon.contains(polygon)

def resolve_overlap(polygon, placed_polygons, main_polygon):
    """Try shifting the polygon to resolve overlap while staying inside the main polygon."""
    for other_polygon in placed_polygons:
        if polygon.intersects(other_polygon):
            # Compute the amount to move
            intersection = polygon.intersection(other_polygon)
            dx = intersection.bounds[2] - intersection.bounds[0]  # Width of intersection
            dy = intersection.bounds[3] - intersection.bounds[1]  # Height of intersection
            
            # Move polygon in a non-colliding direction
            new_polygon_x = translate(polygon, xoff=dx, yoff=0)
            new_polygon_y = translate(polygon, xoff=0, yoff=-dy)
            
            # Prioritize keeping the polygon within bounds
            if is_within_bounds(new_polygon_x, main_polygon) and not any(new_polygon_x.intersects(p) for p in placed_polygons):
                return new_polygon_x
            elif is_within_bounds(new_polygon_y, main_polygon) and not any(new_polygon_y.intersects(p) for p in placed_polygons):
                return new_polygon_y
            else:
                print("Warning: Could not resolve overlap cleanly")
                return None  # Return original if no valid move is found
                
    return polygon  # No overlap found, return unchanged


def update_positions_in_dict(target_zone, count, current_row_y, current_right_x, next_top_left_x):
    zones_data[target_zone]['count'] = count
    zones_data[target_zone]['current_row_y'] = current_row_y
    zones_data[target_zone]['current_right_x'] = current_right_x
    zones_data[target_zone]['next_top_left_x'] = next_top_left_x

def iterate_over_zones_and_place(zones, df, free_space_polygons, placed_modules, crane_access, idx):
    for zone in zones:

            module_coords = place_module_in_zone_horiz_and_verti(row, free_space_polygons[zone], zone, placed_modules, crane_access=crane_access)

            if module_coords:
                df.at[idx, 'coords'] = module_coords
                df.at[idx, 'Placed_Zone'] = zone
                placed_modules[zone].append(module_coords)
                break  # Successfully placed, move to the next module
            else:
                continue
def get_coords_of_each_module(df, free_space_polygons, zones_data):
    placed_modules = {zone: [] for zone in zones_data}  # Dynamically initialize zones
    placed_modules['weighbridge'] = []
    for idx, row in df.iterrows():
        zones, crane_access = get_zones(row['Zone_location'])
        print(f"Processing: {row['Platform']} {row['Module']} -> Potential Zones: {zones}")

        iterate_over_zones_and_place(zones, df, free_space_polygons, placed_modules, crane_access, idx)
        
        if row['Zone_location'] == "Paint":
            other_zone = 'Deck_Interior'
            zones, crane_access = get_zones(other_zone)
            iterate_over_zones_and_place(zones, df, free_space_polygons, placed_modules, crane_access, idx)
        else:  # Only executes if no break occurs (all zones full)
           
            print("All zones full. Stopping placement.")
            break

        print(f"Updated row: {row.to_dict()}")
        print(f"Zones Data: {zones_data}")

    return df, placed_modules

doc = FreeCAD.ActiveDocument  # Ensure your layout is open
date_to_view = input_date()

zones_data = {}

rows = list(string.ascii_uppercase[:8])  # A to H
cols = range(1, 5)  # 1 to 4

for row in rows:
    for col in cols:
        zone_key = f"{row}{col}"
        zones_data[zone_key] = {
            'count': 0,
            'current_row_y': 0,
            'current_right_x': 0,
            'next_top_left_x': 0
        }
# remove ('B4', 'C4', 'F4', 'G3', 'G4', 'H4', 'H3', 'H2') from zones_data.
zones_data = {k: v for k, v in zones_data.items() if k not in ('B4', 'C4', 'F4', 'G3', 'G4', 'H4', 'H3', 'H2')}
#add A0
zones_data['A0'] = {'count': 0, 'current_row_y': 0, 'current_right_x': 0, 'next_top_left_x': 0}
# Step 1: Create Blocks to represent the modules 
file = 'C:/Users/amrin.kareem/Downloads/work/Yard_Optimization/OneDrive_2024-10-16/P_2300_2302_PDM_only.xlsx'
df = get_df(date_to_view, file)
print(df)
# Step 2: Identify Free Spaces
target_polygon_names = ["A1", "A0", "A2", "A3", "A3_2", "B1", "B2", "B3", "B3_2", "C1", "C2", "C3", "D1", "D2", "D3", "D4", "D4_2", "E1", "E2", "E3", "E4", "F1", "F2", "F3", "G1", "G2", "A4", "H1", "weighbridge", "PS1", "PS2", "PS3", "PS4"]
free_space_polygons = compute_free_spaces_from_poly_names(target_polygon_names)
	
# Print the extracted free spaces (Work Out the Logic of coordinates to be placed)
for name, polygon in free_space_polygons.items():
    print(f"Extracted '{name}': {list(polygon.exterior.coords)}")
df, placed_modules = get_coords_of_each_module(df, free_space_polygons, zones_data)
blocks_data = create_block_data(df)

# Step 3: Place modules from DataFrame onto free spaces 
# Place blocks in the free space
place_blocks_in_free_space(blocks_data)

doc.recompute()  # Update the FreeCAD document