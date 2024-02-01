# #-*- encoding:utf -*-
#
# import os
# import cv2
# import torch
# import numpy as np
# from PIL import Image
# import torchvision.transforms as Tvision
# from transformers import AutoProcessor, AutoModelForVision2Seq
#
#
# class KoSmosDetect:
#     def __init__(self, model_meta_file = None):
#
#         self.model_path =model_meta_file if model_meta_file else "microsoft/kosmos-2-patch14-224"
#         self.model = self.__model()
#         self.processor = self.__processor()
#
#     def __model(self):
#         return AutoModelForVision2Seq.from_pretrained(self.model_path)
#
#     def __processor(self):
#         return AutoProcessor.from_pretrained(self.model_path)
#
#     def detect_processor(self, image, simple=True):
#         prompt = "<grounding>An image of" if simple else "<grounding> Describe this image in detail:"
#
#         inputs = self.processor(text=prompt, images=image, return_tensors="pt")
#
#         generated_ids = self.model.generate(
#             pixel_values=inputs["pixel_values"],
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             image_embeds=None,
#             image_embeds_position_mask=inputs["image_embeds_position_mask"],
#             use_cache=True,
#             max_new_tokens=128,
#         )
#         generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#
#         processed_text, entities = self.processor.post_process_generation(generated_text)
#         return processed_text, entities
#
#     def is_overlapping(self, rect1, rect2):
#         x1, y1, x2, y2 = rect1
#         x3, y3, x4, y4 = rect2
#         return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)
#
#     def draw_box(self, image, collect_entity_location):
#         """_summary_
#
#            Args:
#                image (_type_): image or image path
#                collect_entity_location (_type_): _description_
#            """
#         if isinstance(image, Image.Image):
#             image_h = image.height
#             image_w = image.width
#             image = np.array(image)[:, :, [2, 1, 0]]
#         elif isinstance(image, str):
#             if os.path.exists(image):
#                 pil_img = Image.open(image).convert("RGB")
#                 image = np.array(pil_img)[:, :, [2, 1, 0]]
#                 image_h = pil_img.height
#                 image_w = pil_img.width
#             else:
#                 raise ValueError(f"invaild image path, {image}")
#         elif isinstance(image, torch.Tensor):
#             # pdb.set_trace()
#             image_tensor = image.cpu()
#             reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
#             reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
#             image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
#             pil_img = Tvision.ToPILImage()(image_tensor)
#             image_h = pil_img.height
#             image_w = pil_img.width
#             image = np.array(pil_img)[:, :, [2, 1, 0]]
#         else:
#             raise ValueError(f"invaild image format, {type(image)} for {image}")
#
#         if len(collect_entity_location) == 0:
#             return image
#
#         new_image = image.copy()
#         previous_locations = []
#         previous_bboxes = []
#         text_offset = 10
#         text_offset_original = 4
#         text_size = max(0.07 * min(image_h, image_w) / 100, 0.5)
#         text_line = int(max(1 * min(image_h, image_w) / 512, 1))
#         box_line = int(max(2 * min(image_h, image_w) / 512, 2))
#         text_height = text_offset  # init
#         for (phrase, x1_norm, y1_norm, x2_norm, y2_norm) in collect_entity_location:
#             x1, y1, x2, y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(
#                 y2_norm * image_h)
#             # draw bbox
#             # random color
#             color = tuple(np.random.randint(0, 255, size=3).tolist())
#             new_image = cv2.rectangle(new_image, (x1, y1), (x2, y2), color, box_line)
#
#             # add phrase name
#             # decide the text location first
#             for x_prev, y_prev in previous_locations:
#                 if abs(x1 - x_prev) < abs(text_offset) and abs(y1 - y_prev) < abs(text_offset):
#                     y1 += text_height
#
#             if y1 < 2 * text_offset:
#                 y1 += text_offset + text_offset_original
#
#                 # add text background
#             (text_width, text_height), _ = cv2.getTextSize(phrase, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_line)
#             text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - text_height - text_offset_original, x1 + text_width, y1
#
#             for prev_bbox in previous_bboxes:
#                 while self.is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):
#                     text_bg_y1 += text_offset
#                     text_bg_y2 += text_offset
#                     y1 += text_offset
#
#                     if text_bg_y2 >= image_h:
#                         text_bg_y1 = max(0, image_h - text_height - text_offset_original)
#                         text_bg_y2 = image_h
#                         y1 = max(0, image_h - text_height - text_offset_original + text_offset)
#                         break
#
#             alpha = 0.5
#             for i in range(text_bg_y1, text_bg_y2):
#                 for j in range(text_bg_x1, text_bg_x2):
#                     if i < image_h and j < image_w:
#                         new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(color)).astype(np.uint8)
#
#             cv2.putText(
#                 new_image, phrase, (x1, y1 - text_offset_original), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0),
#                 text_line, cv2.LINE_AA
#             )
#             previous_locations.append((x1, y1))
#             previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))
#
#         return new_image
#
#
# if __name__ == "__main__":
#     print("kosmos")