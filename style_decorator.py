import torch
import torch.nn.functional as F

from wct import whitening, coloring

def extract_patches(feature, patch_size, stride):
    ph, pw = patch_size
    sh, sw = stride
    
    # padding the feature
    padh = (ph - 1) // 2
    padw = (pw - 1) // 2
    padding_size = (padw, padw, padh, padh)
    feature = F.pad(feature, padding_size, 'constant', 0)

    # extract patches
    patches = feature.unfold(2, ph, sh).unfold(3, pw, sw)
    patches = patches.contiguous().view(*patches.size()[:-2], -1)
    
    return patches

class StyleDecorator(torch.nn.Module):
    
    def __init__(self):
        super(StyleDecorator, self).__init__()

    def kernel_normalize(self, kernel, k=3):
        b, ch, h, w, kk = kernel.size()
        
        # calc kernel norm
        kernel = kernel.view(b, ch, h*w, kk).transpose(2, 1)
        kernel_norm = torch.norm(kernel.contiguous().view(b, h*w, ch*kk), p=2, dim=2, keepdim=True)
        
        # kernel reshape
        kernel = kernel.view(b, h*w, ch, k, k)
        kernel_norm = kernel_norm.view(b, h*w, 1, 1, 1)
        
        return kernel, kernel_norm

    def conv2d_with_style_kernels(self, features, kernels, patch_size, deconv_flag=False):
        output = list()
        b, c, h, w = features.size()
        
        # padding
        pad = (patch_size - 1) // 2
        padding_size = (pad, pad, pad, pad)
        
        # batch-wise convolutions with style kernels
        for feature, kernel in zip(features, kernels):
            feature = F.pad(feature.unsqueeze(0), padding_size, 'constant', 0)
                
            if deconv_flag:
                padding_size = patch_size - 1
                output.append(F.conv_transpose2d(feature, kernel, padding=padding_size))
            else:
                output.append(F.conv2d(feature, kernel))
        
        return torch.cat(output, dim=0)
        
    def binarize_patch_score(self, features):
        outputs= list()
        
        # batch-wise operation
        for feature in features:
            matching_indices = torch.argmax(feature, dim=0)
            one_hot_mask = torch.zeros_like(feature)

            h, w = matching_indices.size()
            for i in range(h):
                for j in range(w):
                    ind = matching_indices[i, j]
                    one_hot_mask[ind, i, j] = 1
            outputs.append(one_hot_mask.unsqueeze(0))
            
        return torch.cat(outputs, dim=0)
   
    def norm_deconvolution(self, h, w, patch_size):
        mask = torch.ones((h, w))
        fullmask = torch.zeros((h + patch_size - 1, w + patch_size - 1))

        for i in range(patch_size):
            for j in range(patch_size):
                pad = (i, patch_size - i - 1, j, patch_size - j - 1)
                padded_mask = F.pad(mask, pad, 'constant', 0)
                fullmask += padded_mask

        pad_width = (patch_size - 1) // 2
        if pad_width == 0:
            deconv_norm = fullmask
        else:
            deconv_norm = fullmask[pad_width:-pad_width, pad_width:-pad_width]

        return deconv_norm.view(1, 1, h, w)

    def reassemble_feature(self, normalized_content_feature, normalized_style_feature, patch_size, patch_stride):
        # get patches of style feature
        style_kernel = extract_patches(normalized_style_feature, [patch_size, patch_size], [patch_stride, patch_stride])

        # kernel normalize
        style_kernel, kernel_norm = self.kernel_normalize(style_kernel, patch_size)
        
        # convolution with style kernel(patch wise convolution)
        patch_score = self.conv2d_with_style_kernels(normalized_content_feature, style_kernel/kernel_norm, patch_size)
        
        # binarization
        binarized = self.binarize_patch_score(patch_score)
        
        # deconv norm
        deconv_norm = self.norm_deconvolution(h=binarized.size(2), w=binarized.size(3), patch_size=patch_size)

        # deconvolution
        output = self.conv2d_with_style_kernels(binarized, style_kernel, patch_size, deconv_flag=True)
        
        return output/deconv_norm.type_as(output)

    def forward(self, content_feature, style_feature, style_strength=1.0, patch_size=3, patch_stride=1): 
        # 1-1. content feature projection
        normalized_content_feature = whitening(content_feature)

        # 1-2. style feature projection
        normalized_style_feature = whitening(style_feature)

        # 2. swap content and style features
        reassembled_feature = self.reassemble_feature(normalized_content_feature, normalized_style_feature, patch_size=patch_size, patch_stride=patch_stride)

        # 3. reconstruction feature with style mean and covariance matrix
        stylized_feature = coloring(reassembled_feature, style_feature)

        # 4. content and style interpolation
        result_feature = (1 - style_strength) * content_feature + style_strength * stylized_feature
        
        return result_feature
