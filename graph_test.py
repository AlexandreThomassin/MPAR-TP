import pydot
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

graph = pydot.Dot('my_graph', graph_type='graph', bgcolor='white')

# Add nodes
my_node = pydot.Node('S0', label='S0')
graph.add_node(my_node)
# Or, without using an intermediate variable:
graph.add_node(pydot.Node('a', shape='point'))

# Add edges
my_edge = pydot.Edge('S0', 'a', color='blue', weight = 3)
graph.add_edge(my_edge)
# Or, without using an intermediate variable:
graph.add_edge(pydot.Edge('b', 'c', color='blue'))

png_str = graph.create_png(prog='dot')

# treat the DOT output as an image file
sio = io.BytesIO()
sio.write(png_str)
sio.seek(0)
img = mpimg.imread(sio)

# plot the image
imgplot = plt.imshow(img, aspect='equal')
plt.show()