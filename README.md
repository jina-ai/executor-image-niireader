# NiiReader

The `NiiReader` [executor](https://docs.jina.ai/fundamentals/executor/) loads medical images such as fMRI available in a common medical and neuroimaging file format, `NIfTI-1` or `NIfTI-2` with extension `.nii` or `.nii.gz`, into Jina's [`Document`](https://docs.jina.ai/fundamentals/document/) type.
[`NIfTI`](https://nifti.nimh.nih.gov/) files are used very commonly in imaging informatics for neuroscience and even neuroradiology research.
The executor loads the images using the [`Nibabel`](https://nipy.org/nibabel/) library and stores the image in the `blob` attribute of the `Document` as an `ndarray`.
File with extension `.nii` `.nii.gz` is currently supported.



