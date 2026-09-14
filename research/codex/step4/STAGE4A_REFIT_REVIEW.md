# Step-4A refit and qualitative-gate review

## Verified numerical result

Allocated CPU job `4394917` completed with exit `0:0` in 36:46. It used target means and sample standard deviations from all 3,402 mBRSET training images. Parameter selection used 96 BRSET training images, and evaluation used 96 different BRSET training patients. It accessed no mBRSET validation or test image.

On the held-out source patients, the five-statistic standardized squared distance was:

| Transformation | Mean distance over validation draws | Interpretation |
|---|---:|---|
| None | 6.0661 | Original BRSET appearance reference |
| Historical frozen fit | 4.2108 | Earlier parameters improve the narrow objective but leave a large gap |
| New full fit | 0.9018 | Strongest held-out match on the five fitted summaries |
| New overlay-free fit | 0.9421 | Nearly the same narrow match without synthetic spots, holes or halo |

The full fit selected blur 0.5628, illumination strength 0.2936, one bright spot, no dark hole, halo 0.1218, brightness 0.8623, contrast 1.8712, saturation 0.6660 and sensor noise 0.00436. The overlay-free fit selected blur 0.4367, illumination 0.1195, zero spots/holes/halo, brightness 0.7451, contrast 1.8679, saturation 0.5687 and noise 0.00500.

This is a descriptive appearance result. The objective uses only sharpness, brightness, contrast, center-to-edge falloff and saturation means. It does not establish a physical camera model, distribution matching, lesion preservation, clinical realism or diagnostic improvement.

## Qualitative implementation gate

The four-row private gallery was generated from actual training images, with unrelated mBRSET references. It showed no shape, size, crop or channel implementation failure. The target-fitted arms visibly moved BRSET toward a darker, less saturated and higher-contrast appearance. The full arm also introduced bright spots and a broad halo in some examples. Those effects can resemble or obscure retinal findings. The overlay-free arm avoided these synthetic blobs and therefore remains an interpretation-critical control.

The FundusAug component produced severe darkening or bright spots in some displayed draws, as expected from its broad artifact ranges. This comparator is useful for testing generic robustness but must not be described as clinically validated or as the complete GDRNet method.

The private image grid is deliberately excluded from version control because it contains dataset images. Its SHA-256 and provenance are recorded in `preflight.json`.

## Gate decision

Proceed to the corrected one-GPU smoke because the numerical fit generalizes to held-out source patients and the gallery has no implementation failure. Proceed to the seed-0 classifier screen only if the smoke confirms finite optimization, the joint B1 training pool and valid validation inference for all arms. Classifier screening remains validation-only. A promising classifier effect would still require replication and separate diagnostic-preservation validation.
