from plyfile import PlyData, PlyElement
plydata = PlyData.read("output/coffee_martini/frame1/input.ply")
print(plydata.elements[0])