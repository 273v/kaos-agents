# Image fixtures

## Per-file manifest

Required by `docs/oss/50-data-and-fixtures/provenance-policy.md:16`.

| File | Source URL | License | Retrieved | SHA-256 |
|------|------------|---------|-----------|---------|
| iss068e027836-full-moon-south-texas.jpg | https://images-assets.nasa.gov/image/iss068e027836/iss068e027836~orig.jpg | NASA Public Domain (17 USC §105) | 2026-05-11 | 488ee4cc766132b655ff8abff003b5745f3203d85092533ee4ff5870d70d9d3b |

The on-disk file was downsized from the 1.03 MB upstream original
(5568x3712) to ~73 KB (1280x853) via PIL `thumbnail` with
`quality=85`; EXIF bytes were copied through verbatim. The SHA-256
above is for the downsized fixture as committed, not for the upstream
original.

## `iss068e027836-full-moon-south-texas.jpg`

NASA International Space Station photograph (ISS Expedition 68, image
`iss068e027836`): "Full Moon over the South Texas coast", taken by
JAXA astronaut Koichi Wakata aboard the ISS on 2022-12-08.

- **Original source**: <https://images-assets.nasa.gov/image/iss068e027836/iss068e027836~orig.jpg>
- **NASA media page**: <https://images.nasa.gov/details/iss068e027836>
- **Camera**: Nikon D5 (EXIF preserved)
- **Original size**: 5568x3712, 1.03 MB
- **Fixture size**: 1280x853, ~73 KB (downsized via PIL `thumbnail` with
  `quality=85`; EXIF bytes copied through verbatim so EXIF assertions
  still pass)
- **EXIF surface**: includes `Make=NIKON CORPORATION`, `Model=NIKON D5`,
  `DateTimeOriginal=2022:12:08 15:36:04`, `Software=Adobe Photoshop 23.4`,
  `ImageDescription=GMT342_17_30_Koichi Wakata_1162_Full Moon and South
  Texas coast` plus 50+ other tags.

### License (NASA Media Usage Guidelines, public domain)

> NASA still images, audio files, video, and computer files used in the
> rendition of 3-dimensional models, such as texture maps and polygon
> data in any format, generally are not subject to copyright in the
> United States. You may use this material for educational or
> informational purposes, including photo collections, textbooks,
> public exhibits, computer graphical simulations and Internet
> Web pages. This general permission extends to personal Web pages.

See <https://www.nasa.gov/nasa-brand-center/images-and-media/> for the
full media-usage policy. NASA imagery is treated as Creative Commons
Public Domain (CC-PD) for redistribution.

### Why this fixture?

The KAOS test policy bars fake byte-string fixtures
(`b"hello"` as an "image"). This file gives the test suite a real
photograph with:

1. Genuine EXIF metadata (Nikon D5 capture + Photoshop post-processing
   chain), so the `kaos-source` EXIF extractor exercises the real
   sub-IFD walk rather than the empty-EXIF code path.
2. A subject any vision-capable LLM can identify ("Earth from orbit,
   moon visible") so the VLM describe path produces a verifiable answer.

The image was downsized from 1.03 MB to ~75 KB to keep the repo small;
EXIF bytes are copied through verbatim by `PIL.Image.save(..., exif=...)`
so every tag listed above is still present after the resize. Verified
via `kaos_source.parsers.metadata.image.extract_image_metadata`.
