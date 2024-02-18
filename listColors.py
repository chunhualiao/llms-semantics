import webcolors

def list_css3_color_names_and_rgb():
    """
    List CSS3 color names and their matching RGB values.
    
    Returns:
    - A dictionary with color names as keys and their RGB values as values.
    """
    color_names_to_rgb = {name: webcolors.name_to_rgb(name) for name in webcolors.CSS3_NAMES_TO_HEX}
    return color_names_to_rgb

# Get the list of color names with matching RGB codes
color_names_rgb = list_css3_color_names_and_rgb()

# Print the color names with their RGB values
for name, rgb in color_names_rgb.items():
    print(f"'{name}'")
#    print(f"{name}: RGB{rgb}")

