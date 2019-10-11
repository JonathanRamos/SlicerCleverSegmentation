import os
import vtk, qt, ctk, slicer
import logging
from SegmentEditorEffects import *

class SegmentEditorEffect(AbstractScriptedSegmentEditorAutoCompleteEffect):
  """ AutoCompleteEffect is an effect that can create a full segmentation
      from a partial segmentation (not all slices are segmented or only
      part of the target structures are painted).
  """

  def __init__(self, scriptedEffect):
    AbstractScriptedSegmentEditorAutoCompleteEffect.__init__(self, scriptedEffect)
    scriptedEffect.name = 'CleverSegmentation'
    self.minimumNumberOfSegments = 2
    self.clippedMasterImageDataRequired = True # master volume intensities are used by this effect
    self.clippedMaskImageDataRequired = True # masking is used
    self.cleverSegFilter = None

  def clone(self):
    import qSlicerSegmentationsEditorEffectsPythonQt as effects
    clonedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
    clonedEffect.setPythonSource(__file__.replace('\\','/'))
    return clonedEffect

  def icon(self):
    iconPath = os.path.join(os.path.dirname(__file__), 'SegmentEditorEffect.png')
    if os.path.exists(iconPath):
      return qt.QIcon(iconPath)
    return qt.QIcon()

  def helpText(self):
    return """<html>Growing segments to create complete segmentation<br>.
Location, size, and shape of initial segments and content of master volume are taken into account.
Final segment boundaries will be placed where master volume brightness changes abruptly. Instructions:<p>
<ul style="margin: 0">
<li>Use Paint or other offects to draw seeds in each region that should belong to a separate segment.
Paint each seed with a different segment. Minimum two segments are required.</li>
<li>Click <dfn>Initialize</dfn> to compute preview of full segmentation.</li>
<li>Browse through image slices. If previewed segmentation result is not correct then switch to
Paint or other effects and add more seeds in the misclassified region. Full segmentation will be
updated automatically within a few seconds</li>
<li>Click <dfn>Apply</dfn> to update segmentation with the previewed result.</li>
</ul><p>
If segments overlap, segment higher in the segments table will have priority.
The effect uses fast clever-segmentation method</a>.
<p></html>"""


  def reset(self):
    self.cleverSegFilter = None
    AbstractScriptedSegmentEditorAutoCompleteEffect.reset(self)
    self.updateGUIFromMRML()

  def computePreviewLabelmap(self, mergedImage, outputLabelmap):
    import vtkSlicerModuleLogicPython as vtkCleverSeg

    if not self.cleverSegFilter:
      self.cleverSegFilter = vtkCleverSeg.vtkImageCleverSegSegment()
      self.cleverSegFilter.SetIntensityVolume(self.clippedMasterImageData)
      self.cleverSegFilter.SetMaskVolume(self.clippedMaskImageData)
      maskExtent = self.clippedMaskImageData.GetExtent() if self.clippedMaskImageData else None
      if maskExtent is not None and maskExtent[0] <= maskExtent[1] and maskExtent[2] <= maskExtent[3] and maskExtent[4] <= maskExtent[5]:
        # Mask is used.
        # Grow the extent more, as background segment does not surround region of interest.
        self.extentGrowthRatio = 0.50
      else:
        # No masking is used.
        # Background segment is expected to surround region of interest, so narrower margin is enough.
        self.extentGrowthRatio = 0.20

      self.extentGrowthRatio

    self.cleverSegFilter.SetSeedLabelVolume(mergedImage)
    self.cleverSegFilter.Update()

    outputLabelmap.DeepCopy( self.cleverSegFilter.GetOutput() )
