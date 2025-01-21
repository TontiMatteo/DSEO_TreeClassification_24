Google earth engine code for 3 months: 
https://code.earthengine.google.com/da938139fd0af01811c57578fd379d3d

problem is probably at the end:
i used sampleRegions() but it gives us less rows for 3 months

for 3 months: april june october
 // Sample the image (neighborImg) for each species point
    var samples = neighborImg.sampleRegions({
    collection: speciesCollection,  // Use the points from speciesCollection
    scale: 10                      // Specify the scale of 10 meters per pixel
    });
we get : Final dimensions of the GeoDataFrame: (14959, 26)

only april gives us:
Final dimensions of the GeoDataFrame: (32423, 12)


when we use ee.Reducer.tolist() we get 37907 rows, but it also seems weird
 //chatgpt also suggests this: 
    // var samples = neighborImg.reduceRegions({
    //collection: speciesCollection,
    //reducer: ee.Reducer.toList(),
    //scale: 10
    //});
Final dimensions of the GeoDataFrame: (37907, 26)

one more problem is also that ee.Reducer.tolist() gives us empty lists [] for patches which did not pass the cloud filter. When trying to do the 4D numpy array it says ValueError: all input arrays must have the same shape

