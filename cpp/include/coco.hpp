// C26 — coco.hpp: YOLO-80 -> COCO-91 ids, image ids from filenames, COCO results JSON
#pragma once
#include "postprocess.hpp"
#include <ostream>
#include <string>
#include <vector>

int coco80_to_coco91(int cls80);                    // throws std::out_of_range outside 0..79
long long coco_image_id(const std::string& path);   // ".../000000397133.jpg" -> 397133, non-numeric -> -1

// Appends dets in COCO results format: {"image_id","category_id","bbox":[x,y,w,h],"score"}.
// `first` tracks comma placement across calls.
void append_coco_json(std::ostream& os, long long image_id, const std::vector<Det>& dets, bool& first);
