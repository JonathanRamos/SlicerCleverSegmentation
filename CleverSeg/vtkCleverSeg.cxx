#include "vtkCleverSeg.h"

#include <iostream>
#include <vector>

#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkLoggingMacros.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkStreamingDemandDrivenPipeline.h>
#include <vtkTimerLog.h>

vtkStandardNewMacro(vtkCleverSeg);

//----------------------------------------------------------------------------
typedef float DistancePixelType;  // type for cost function
const int DistancePixelTypeID = VTK_FLOAT;

typedef unsigned char MaskPixelType;
const int MaskPixelTypeID = VTK_UNSIGNED_CHAR;

const DistancePixelType DIST_INF = 1.0;//std::numeric_limits<DistancePixelType>::max();
const DistancePixelType DIST_EPSILON = 0.0;//1e-3;

//----------------------------------------------------------------------------
class vtkCleverSeg::vtkInternal
{
public:
	vtkInternal();
	virtual ~vtkInternal();

	void Reset();

	template<typename IntensityPixelType, typename LabelPixelType>
	bool InitializationAHP(vtkImageData *intensityVolume, vtkImageData *seedLabelVolume, vtkImageData *maskLabelVolume);

	template<typename IntensityPixelType, typename LabelPixelType>
	void DijkstraBasedClassificationAHP(vtkImageData *intensityVolume, vtkImageData *seedLabelVolume, vtkImageData *maskLabelVolume);

	template <class SourceVolType>
	bool ExecuteCleverSeg(vtkImageData *intensityVolume, vtkImageData *seedLabelVolume, vtkImageData *maskLabelVolume, vtkImageData *resultLabelVolume);

	template< class SourceVolType, class SeedVolType>
	bool ExecuteCleverSeg2(vtkImageData *intensityVolume, vtkImageData *seedLabelVolume, vtkImageData *maskLabelVolume);

	// Stores the shortest distance from known labels to each point
	// If a point is set to DIST_INF then that point will modified, as a shorter distance path will be found.
	// If a point is set to DIST_EPSILON, then the distance is so small that a shorter path will not be found and so
	// the point will not be relabeled.
	vtkSmartPointer<vtkImageData> m_DistanceVolume;

	// Resulting segmentation
	vtkSmartPointer<vtkImageData> m_ResultLabelVolume;

	// Distance and labeling result volume in the previous step
	vtkSmartPointer<vtkImageData> m_DistanceVolumePre;
	vtkSmartPointer<vtkImageData> m_ResultLabelVolumePre;

	long m_DimX;
	long m_DimY;
	long m_DimZ;
	float maxC;
	std::vector<long> m_NeighborIndexOffsets;
	std::vector<unsigned char> m_NumberOfNeighbors; // size of neighborhood (everywhere the same except at the image boundary)

	bool m_bSegInitialized;
};

//-----------------------------------------------------------------------------
vtkCleverSeg::vtkInternal::vtkInternal()
{
	m_bSegInitialized = false;
	m_DistanceVolume = vtkSmartPointer<vtkImageData>::New();
	m_DistanceVolumePre = vtkSmartPointer<vtkImageData>::New();
	m_ResultLabelVolume = vtkSmartPointer<vtkImageData>::New();
	m_ResultLabelVolumePre = vtkSmartPointer<vtkImageData>::New();
};

//-----------------------------------------------------------------------------
vtkCleverSeg::vtkInternal::~vtkInternal()
{
	this->Reset();
};

//-----------------------------------------------------------------------------
void vtkCleverSeg::vtkInternal::Reset()
{
	m_bSegInitialized = false;
	m_DistanceVolume->Initialize();
	m_DistanceVolumePre->Initialize();
	m_ResultLabelVolume->Initialize();
	m_ResultLabelVolumePre->Initialize();
}

//-----------------------------------------------------------------------------
template<typename IntensityPixelType, typename LabelPixelType>
bool vtkCleverSeg::vtkInternal::InitializationAHP(
	vtkImageData *intensityVolume,
	vtkImageData *seedLabelVolume,
	vtkImageData *maskLabelVolume)
{

	long dimXYZ = m_DimX * m_DimY * m_DimZ;
	maxC = 0;
	LabelPixelType* seedLabelVolumePtr = static_cast<LabelPixelType*>(seedLabelVolume->GetScalarPointer());
    IntensityPixelType* imSrc = static_cast<IntensityPixelType*>(intensityVolume->GetScalarPointer());
	MaskPixelType* maskLabelVolumePtr = nullptr;
	if (seedLabelVolume != nullptr)
	{
		maskLabelVolumePtr = static_cast<MaskPixelType*>(maskLabelVolume->GetScalarPointer());
	}

	if (!m_bSegInitialized)
	{
		m_ResultLabelVolume->SetOrigin(seedLabelVolume->GetOrigin());
		m_ResultLabelVolume->SetSpacing(seedLabelVolume->GetSpacing());
		m_ResultLabelVolume->SetExtent(seedLabelVolume->GetExtent());
		m_ResultLabelVolume->AllocateScalars(seedLabelVolume->GetScalarType(), 1);
		m_DistanceVolume->SetOrigin(seedLabelVolume->GetOrigin());
		m_DistanceVolume->SetSpacing(seedLabelVolume->GetSpacing());
		m_DistanceVolume->SetExtent(seedLabelVolume->GetExtent());
		m_DistanceVolume->AllocateScalars(DistancePixelTypeID, 1);
		m_ResultLabelVolumePre->SetExtent(0, -1, 0, -1, 0, -1);
		m_ResultLabelVolumePre->AllocateScalars(seedLabelVolume->GetScalarType(), 1);
		m_DistanceVolumePre->SetExtent(0, -1, 0, -1, 0, -1);
		m_DistanceVolumePre->AllocateScalars(DistancePixelTypeID, 1);
		LabelPixelType* resultLabelVolumePtr = static_cast<LabelPixelType*>(m_ResultLabelVolume->GetScalarPointer());
		DistancePixelType* distanceVolumePtr = static_cast<DistancePixelType*>(m_DistanceVolume->GetScalarPointer());

		// Compute index offset
		m_NeighborIndexOffsets.clear();
		// Neighbors are traversed in the order of m_NeighborIndexOffsets,
		// therefore one would expect that the offsets should
		// be as continuous as possible (e.g., x coordinate
		// should change most quickly), but that resulted in
		// about 5-6% longer computation time. Therefore,
		// we put indices in order x1y1z1, x1y1z2, x1y1z3, etc.
		for (int ix = -1; ix <= 1; ix++)
		{
			for (int iy = -1; iy <= 1; iy++)
			{
				for (int iz = -1; iz <= 1; iz++)
				{
					if (ix == 0 && iy == 0 && iz == 0)
					{
						continue;
					}
					m_NeighborIndexOffsets.push_back(long(ix) + m_DimX * (long(iy) + m_DimY * long(iz)));
				}
			}
		}

		// Determine neighborhood size for computation at each voxel.
		// The neighborhood size is everywhere the same (size of m_NeighborIndexOffsets)
		// except at the edges of the volume, where the neighborhood size is 0.
		m_NumberOfNeighbors.resize(dimXYZ);
		const unsigned char numberOfNeighbors = m_NeighborIndexOffsets.size();
		unsigned char* nbSizePtr = &(m_NumberOfNeighbors[0]);
		for (int z = 0; z < m_DimZ; z++)
		{
			bool zEdge = (z == 0 || z == m_DimZ - 1);
			for (int y = 0; y < m_DimY; y++)
			{
				bool yEdge = (y == 0 || y == m_DimY - 1);
				*(nbSizePtr++) = 0; // x == 0 (there is always padding, so we don'neighborNewDistance need to check if m_DimX>0)
				unsigned char nbSize = (zEdge || yEdge) ? 0 : numberOfNeighbors;
				for (int x = m_DimX - 2; x > 0; x--)
				{
					*(nbSizePtr++) = nbSize;
				}
				*(nbSizePtr++) = 0; // x == m_DimX-1 (there is always padding, so we don'neighborNewDistance need to check if m_DimX>1)
			}
		}

		if (!maskLabelVolumePtr)
		{
			// no mask
			for (long index = 0; index < dimXYZ; index++)
			{
				LabelPixelType seedValue = seedLabelVolumePtr[index];
				resultLabelVolumePtr[index] = seedValue;
				if (seedValue == 0) {
					distanceVolumePtr[index] = DIST_EPSILON;
				}
				else {
					distanceVolumePtr[index] = DIST_INF;
				}
                if (imSrc[index] > maxC)
                {
				    maxC = imSrc[index];
                }
			}
			
		}
		else
		{
			// with mask
			for (long index = 0; index < dimXYZ; index++)
			{
				if (maskLabelVolumePtr[index] != 0)
				{
					// masked region
					// small distance will prevent overwriting of masked voxels
					distanceVolumePtr[index] = DIST_INF;
					// we don't add masked voxels to the heap
					// to exclude them from region growing
				}
				else {
					// non-masked region
					LabelPixelType seedValue = seedLabelVolumePtr[index];
					resultLabelVolumePtr[index] = seedValue;
					if (seedValue == 0)
					{
						distanceVolumePtr[index] = DIST_EPSILON;
					}
					else {
                        distanceVolumePtr[index] = DIST_INF;
                    }
				}
				
                if (imSrc[index] > maxC)
                {
				    maxC = imSrc[index];
                }
			}
		}
    } else {
		// Already initialized
		LabelPixelType* resultLabelVolumePtr = static_cast<LabelPixelType*>(m_ResultLabelVolume->GetScalarPointer());
		DistancePixelType* distanceVolumePtr = static_cast<DistancePixelType*>(m_DistanceVolume->GetScalarPointer());

		for (long index = 0; index < dimXYZ; index++)
		{
			if (seedLabelVolumePtr[index] != 0)
			{
				// Only grow from new/changed seeds
				if (resultLabelVolumePtr[index] != seedLabelVolumePtr[index])
				{
					resultLabelVolumePtr[index] = seedLabelVolumePtr[index];
				}
			} else {
				distanceVolumePtr[index] = DIST_INF;
				resultLabelVolumePtr[index] = 0;
			}
		}
	}
	return true;
}

//-----------------------------------------------------------------------------
template<typename IntensityPixelType, typename LabelPixelType>
void vtkCleverSeg::vtkInternal::DijkstraBasedClassificationAHP(
	vtkImageData *intensityVolume,
	vtkImageData *vtkNotUsed(seedLabelVolume),
	vtkImageData *vtkNotUsed(maskLabelVolume))
{
	LabelPixelType* resultLabelVolumePtr = static_cast<LabelPixelType*>(m_ResultLabelVolume->GetScalarPointer());
	IntensityPixelType* imSrc = static_cast<IntensityPixelType*>(intensityVolume->GetScalarPointer());
    long dimXYZ = m_DimX * m_DimY * m_DimZ;

	if (!m_bSegInitialized)
	{
		// Full computation
		DistancePixelType* distanceVolumePtr = static_cast<DistancePixelType*>(m_DistanceVolume->GetScalarPointer());
		LabelPixelType* resultLabelVolumePtr = static_cast<LabelPixelType*>(m_ResultLabelVolume->GetScalarPointer());

		float voxDiff, strength, weightDiff = 0;
		const float theta = 0.01;
		long idxCenter, idxNgbh, cont = 0;
		const long maxIt = 999;
		bool converged = false;
		std::vector<bool> visited = std::vector<bool>(dimXYZ, false);

		while (!converged) {
			converged = true;
			for (idxCenter = 0; idxCenter < dimXYZ; idxCenter++) { // for each voxel
				if (resultLabelVolumePtr[idxCenter] != 0 && !visited[idxCenter]) { // Dont expand unlabelled voxel
					visited[idxCenter] = true;
					unsigned char nbSize = m_NumberOfNeighbors[idxCenter]; // get number of neighbours
					for (unsigned int i = 0; i < nbSize; i++) { // For each one of the neighbours 
						idxNgbh = idxCenter + m_NeighborIndexOffsets[i]; // calculate neighbour index

						voxDiff = (maxC - fabs(imSrc[idxCenter] - imSrc[idxNgbh])) / maxC; // calcule shifted/normalized voxel intensity difference
						strength = (float)(voxDiff * distanceVolumePtr[idxCenter]); // calculate a new strength
						weightDiff = (float)strength - distanceVolumePtr[idxNgbh];

						if (weightDiff > theta) { // wont average if it only changes above the third/fourth decimal place
							visited[idxNgbh] = false;
							distanceVolumePtr[idxNgbh] = (distanceVolumePtr[idxNgbh] + strength + distanceVolumePtr[idxCenter]) / 3; // update weight
							resultLabelVolumePtr[idxNgbh] = resultLabelVolumePtr[idxCenter]; // update label
							converged = false; // Keep iterating
						}
					}
				}
			}
			cont++;
			if (cont == maxIt) break; // probably will never reach max, just in case
		}
		//std::cout << "Iterations " << cont << endl;

	}
	else
	{
		// Quick update

		LabelPixelType* resultLabelVolumePrePtr = static_cast<LabelPixelType*>(m_ResultLabelVolumePre->GetScalarPointer());
		DistancePixelType* distanceVolumePtr = static_cast<DistancePixelType*>(m_DistanceVolume->GetScalarPointer());

		// Adaptive Dijkstra
		long dimXYZ = m_DimX * m_DimY * m_DimZ;

		float voxDiff, strength, weightDiff = 0;
		const float theta = 0.01;
		long idxCenter, idxNgbh, i, cont = 0;
		const long maxIt = 999;
		bool converged = false;
		std::vector<bool> visited = std::vector<bool>(dimXYZ, false);

		while (!converged) {
			converged = true;
			for (idxCenter = 0; idxCenter < dimXYZ; idxCenter++) { // for each voxel
				if (resultLabelVolumePtr[idxCenter] != 0 && !visited[idxCenter]) { // Dont expand unlabelled voxel
					visited[idxCenter] = true;
					unsigned char nbSize = m_NumberOfNeighbors[idxCenter]; // get number of neighbours
					for (unsigned int i = 0; i < nbSize; i++) { // For each one of the neighbours 
						idxNgbh = idxCenter + m_NeighborIndexOffsets[i]; // calculate neighbour index

						voxDiff = (maxC - fabs(imSrc[idxCenter] - imSrc[idxNgbh])) / maxC; // calcule shifted/normalized voxel intensity difference
						strength = (float)(voxDiff * distanceVolumePtr[idxCenter]); // calculate a new strength
						weightDiff = (float)strength - distanceVolumePtr[idxNgbh];

						if (weightDiff > theta) { // wont average if it only changes above the third/fourth decimal place
							visited[idxNgbh] = false;
							distanceVolumePtr[idxNgbh] = (distanceVolumePtr[idxNgbh] + strength + distanceVolumePtr[idxCenter]) / 3; // update weight
							resultLabelVolumePtr[idxNgbh] = resultLabelVolumePtr[idxCenter]; // update label
							converged = false; // Keep iterating
						}
					}
				}
			}
			cont++;
			if (cont == maxIt) break; // probably will never reach max, just in case
		}
		//std::cout << "Iterations " << cont << endl;


		// Update previous labels and distance information
		m_ResultLabelVolumePre->DeepCopy(m_ResultLabelVolume);
		m_DistanceVolumePre->DeepCopy(m_DistanceVolume);
		m_bSegInitialized = true;
	}

}

//-----------------------------------------------------------------------------
template< class IntensityPixelType, class LabelPixelType>
bool vtkCleverSeg::vtkInternal::ExecuteCleverSeg2(vtkImageData *intensityVolume, vtkImageData *seedLabelVolume, vtkImageData *maskLabelVolume)
{
	int* imSize = intensityVolume->GetDimensions();
	m_DimX = imSize[0];
	m_DimY = imSize[1];
	m_DimZ = imSize[2];

	if (m_DimX <= 2 || m_DimY <= 2 || m_DimZ <= 2)
	{
		// image is too small (there should be space for at least one voxel padding around the image)
		vtkGenericWarningMacro("vtkCleverSeg: image size is too small");
		return false;
	}

	if (!InitializationAHP<IntensityPixelType, LabelPixelType>(intensityVolume, seedLabelVolume, maskLabelVolume))
	{
		return false;
	}

	DijkstraBasedClassificationAHP<IntensityPixelType, LabelPixelType>(intensityVolume, seedLabelVolume, maskLabelVolume);
	return true;
}

//----------------------------------------------------------------------------
template <class SourceVolType>
bool vtkCleverSeg::vtkInternal::ExecuteCleverSeg(vtkImageData *intensityVolume, vtkImageData *seedLabelVolume,
	vtkImageData *maskLabelVolume, vtkImageData *resultLabelVolume)
{
	int* extent = intensityVolume->GetExtent();
	double* spacing = intensityVolume->GetSpacing();
	double* origin = intensityVolume->GetOrigin();
	int* seedExtent = seedLabelVolume->GetExtent();
	double* seedSpacing = seedLabelVolume->GetSpacing();
	double* seedOrigin = seedLabelVolume->GetOrigin();
	const double compareTolerance = (spacing[0] + spacing[1] + spacing[2]) / 3.0 * 0.01;

	// Return with error if intensity volume geometry differs from seed label volume geometry
	if (seedExtent[0] != extent[0] || seedExtent[1] != extent[1]
		|| seedExtent[2] != extent[2] || seedExtent[3] != extent[3]
		|| seedExtent[4] != extent[4] || seedExtent[5] != extent[5]
		|| fabs(seedOrigin[0] - origin[0]) > compareTolerance
		|| fabs(seedOrigin[1] - origin[1]) > compareTolerance
		|| fabs(seedOrigin[2] - origin[2]) > compareTolerance
		|| fabs(seedSpacing[0] - spacing[0]) > compareTolerance
		|| fabs(seedSpacing[1] - spacing[1]) > compareTolerance
		|| fabs(seedSpacing[2] - spacing[2]) > compareTolerance)
	{
		vtkGenericWarningMacro("vtkCleverSeg: Seed label volume geometry does not match intensity volume geometry");
		return false;
	}

	// Return with error if intensity volume geometry differs from mask label volume geometry
	if (maskLabelVolume)
	{
		int* maskExtent = maskLabelVolume->GetExtent();
		double* maskSpacing = maskLabelVolume->GetSpacing();
		double* maskOrigin = maskLabelVolume->GetOrigin();
		if (maskExtent[0] != extent[0] || maskExtent[1] != extent[1]
			|| maskExtent[2] != extent[2] || maskExtent[3] != extent[3]
			|| maskExtent[4] != extent[4] || maskExtent[5] != extent[5]
			|| fabs(maskOrigin[0] - origin[0]) > compareTolerance
			|| fabs(maskOrigin[1] - origin[1]) > compareTolerance
			|| fabs(maskOrigin[2] - origin[2]) > compareTolerance
			|| fabs(maskSpacing[0] - spacing[0]) > compareTolerance
			|| fabs(maskSpacing[1] - spacing[1]) > compareTolerance
			|| fabs(maskSpacing[2] - spacing[2]) > compareTolerance)
		{
			vtkGenericWarningMacro("vtkCleverSeg: Mask label volume geometry does not match intensity volume geometry");
			return false;
		}
		if (maskLabelVolume->GetScalarType() != MaskPixelTypeID || maskLabelVolume->GetNumberOfScalarComponents() != 1)
		{
			vtkGenericWarningMacro("vtkCleverSeg: Mask label volume scalar must be single-component unsigned char");
			return false;
		}
	}

	// Restart cleverseg from scratch if image size is changed (then cached buffers cannot be reused)
	int* outExtent = m_ResultLabelVolume->GetExtent();
	double* outSpacing = m_ResultLabelVolume->GetSpacing();
	double* outOrigin = m_ResultLabelVolume->GetOrigin();
	if (outExtent[0] != extent[0] || outExtent[1] != extent[1]
		|| outExtent[2] != extent[2] || outExtent[3] != extent[3]
		|| outExtent[4] != extent[4] || outExtent[5] != extent[5]
		|| fabs(outOrigin[0] - origin[0]) > compareTolerance
		|| fabs(outOrigin[1] - origin[1]) > compareTolerance
		|| fabs(outOrigin[2] - origin[2]) > compareTolerance
		|| fabs(outSpacing[0] - spacing[0]) > compareTolerance
		|| fabs(outSpacing[1] - spacing[1]) > compareTolerance
		|| fabs(outSpacing[2] - spacing[2]) > compareTolerance)
	{
		this->Reset();
	}
	else if (m_ResultLabelVolume->GetScalarType() != seedLabelVolume->GetScalarType())
	{
		this->Reset();
	}

	bool success = false;
	switch (seedLabelVolume->GetScalarType())
	{
		vtkTemplateMacro((success = ExecuteCleverSeg2<SourceVolType, VTK_TT>(intensityVolume, seedLabelVolume, maskLabelVolume)));
	default:
		vtkGenericWarningMacro("vtkOrientedImageDataResample::MergeImage: Unknown ScalarType");
	}

	if (success)
	{
		resultLabelVolume->ShallowCopy(this->m_ResultLabelVolume);
	}
	else
	{
		resultLabelVolume->Initialize();
	}
	return success;
}

//-----------------------------------------------------------------------------
vtkCleverSeg::vtkCleverSeg()
{
	this->Internal = new vtkInternal();
	this->SetNumberOfInputPorts(3);
	this->SetNumberOfOutputPorts(1);
}

//-----------------------------------------------------------------------------
vtkCleverSeg::~vtkCleverSeg()
{
	delete this->Internal;
}

//-----------------------------------------------------------------------------
void vtkCleverSeg::ExecuteDataWithInformation(
	vtkDataObject *resultLabelVolumeDataObject, vtkInformation* vtkNotUsed(resultLabelVolumeInfo))
{
	vtkImageData *intensityVolume = vtkImageData::SafeDownCast(this->GetInput(0));
	vtkImageData *seedLabelVolume = vtkImageData::SafeDownCast(this->GetInput(1));
	vtkImageData *maskLabelVolume = vtkImageData::SafeDownCast(this->GetInput(2));
	vtkImageData *resultLabelVolume = vtkImageData::SafeDownCast(resultLabelVolumeDataObject);

	vtkNew<vtkTimerLog> logger;
	logger->StartTimer();

	switch (intensityVolume->GetScalarType())
	{
		vtkTemplateMacro(this->Internal->ExecuteCleverSeg<VTK_TT>(intensityVolume, seedLabelVolume, maskLabelVolume, resultLabelVolume));
		break;
	}
	logger->StopTimer();
	vtkDebugMacro(<< "vtkCleverSeg execution time: " << logger->GetElapsedTime());
}

//-----------------------------------------------------------------------------
int vtkCleverSeg::RequestInformation(
	vtkInformation * request,
	vtkInformationVector **inputVector,
	vtkInformationVector *outputVector)
{
	// get the info objects
	vtkInformation *inInfo = inputVector[0]->GetInformationObject(1);
	if (inInfo != nullptr)
	{
		this->Superclass::RequestInformation(request, inputVector, outputVector);
	}
	return 1;
}

//-----------------------------------------------------------------------------
void vtkCleverSeg::Reset()
{
	this->Internal->Reset();
}

//-----------------------------------------------------------------------------
void vtkCleverSeg::PrintSelf(ostream &os, vtkIndent indent)
{
	// XXX Implement this function
	this->Superclass::PrintSelf(os, indent);
}