cmake_minimum_required(VERSION 3.13.4)

project(CleverSeg)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/lassoan/SlicerCleverSegmentation#cleversegmentation")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Jonathan Ramos (University of São Paulo)")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/lassoan/SlicerCleverSegmentation/master/CleverSegmentation/Resources/Icons/CleverSegmentation.png")
set(EXTENSION_SCREENSHOTURLS "https://user-images.githubusercontent.com/3834596/66679138-d7dc1700-ec43-11e9-8c36-51976a1121d3.png" "https://user-images.githubusercontent.com/3834596/66679174-f215f500-ec43-11e9-84f1-85f8b5c4d568.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(CleverSegmentation)
add_subdirectory(CleverSeg)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
