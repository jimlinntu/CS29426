class Image:
    def __init__(self, img, mask, origin):
        '''
          origin (might not be (0, 0))
            -------------------->
            |
            |       img, mask
            |
            |
            v
        '''
        assert len(img.shape) == 3
        assert len(mask.shape) == 3 and mask.shape[2] == 1
        assert origin.shape == (2, ) and origin.dtype == np.int32

        self.img = img.astype(np.int32)
        self.mask = mask
        self.origin = origin
        self.h = img.shape[0]
        self.w = img.shape[1]
        self.bot_right = origin + np.array([self.w-1, self.h-1], dtype=np.int32)

    def merge(self, image2):
        assert isinstance(image2, Image)
        # determine the new origin, the merged size

        merged_origin = np.minimum(self.origin, image2.origin)
        merged_bot_right = np.maximum(self.bot_right, image2.bot_right)

        # NOTE: +1 is important because boundaries are inclusive
        merged_w, merged_h = (merged_bot_right - merged_origin + 1).astype(np.int32)

        # Merge two images according to their mask
        canvas = np.zeros((merged_h, merged_w, 3), dtype=np.float32)
        mask = np.zeros((merged_h, merged_w, 3), dtype=np.float32)

        canvas2 = np.zeros((merged_h, merged_w, 3), dtype=np.float32)
        mask2 = np.zeros((merged_h, merged_w, 3), dtype=np.float32)

        out = np.zeros((merged_h, merged_w, 3), dtype=np.float32)

        # Paste the first image under the new coordinate system
        x, y = self.origin - merged_origin
        canvas[y:y+self.h, x:x+self.w, :] = self.img
        # mask broadcasting
        mask[y:y+self.h, x:x+self.w, :] = self.mask

        x2, y2 = image2.origin - merged_origin
        canvas2[y2:y2+image2.h, x2:x2+image2.w, :] = image2.img
        # mask broadcasting
        mask2[y2:y2+image2.h, x2:x2+image2.w, :] = image2.mask

        # Weighted average between two images intersection
        select = (mask > 0) & (mask2 > 0)
        out[select] = \
            (mask[select]*canvas[select] + mask2[select] * canvas2[select]) / (mask[select] + mask2[select])

        # Only one part exists
        select = ((mask > 0) ^ ((mask > 0) & (mask2 > 0))).astype(np.bool)
        out[select] = canvas[select]

        select = ((mask2 > 0) ^ ((mask > 0) & (mask2 > 0))).astype(np.bool)
        out[select] = canvas2[select]

        out = np.clip(out, 0, 255).astype(np.int32)
        return Image(out, (mask+mask2)[:, :, 0:1], merged_origin)

    def write(self, path, jpg_quality):
        cv2.imwrite(path, self.img, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
