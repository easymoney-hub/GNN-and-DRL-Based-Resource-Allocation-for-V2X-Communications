import matplotlib.pyplot as plt

# Define the coordinates for the lanes
up_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]
down_lanes = [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2, 750 - 3.5 / 2]
left_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]
right_lanes = [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2, 1299 - 3.5 / 2]

# Create the plot with correct lane directions based on your clarification
plt.figure(figsize=(8, 8))

# Set width and height limits
width = 750
height = 1299

# Plot the lanes with corrected directions
for i in range(len(up_lanes)):
    # Up and Down lanes: horizontal lines (up is positive, down is negative)
    plt.plot([up_lanes[i], up_lanes[i]], [0, height],color='blue')  # Up lanes
    plt.plot([down_lanes[i], down_lanes[i]],[0, height],  color='red')  # Down lanes

    # Left and Right lanes: vertical lines (left is negative, right is positive)
    plt.plot([0, width],[left_lanes[i], left_lanes[i]],  color='green')  # Left lanes
    plt.plot([0, width],[right_lanes[i], right_lanes[i]],  color='purple')  # Right lanes

# Add labels and title
plt.title('Grid Lane Design (Corrected Directions)')
plt.xlabel('X')
plt.ylabel('Y')

# Set limits for better visualization
plt.xlim(-100, 1500)
plt.ylim(-100, 1500)

# Add a legend
plt.legend(['Up Lane', 'Down Lane', 'Left Lane', 'Right Lane'])

# Show the plot
plt.grid(True)
plt.show()
