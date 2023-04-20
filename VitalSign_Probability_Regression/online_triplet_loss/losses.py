# AUTOGENERATED! DO NOT EDIT! File to edit: triplet_loss.ipynb (unless otherwise specified).

__all__ = ['batch_hard_triplet_loss', 'batch_all_triplet_loss']

# Cell
import torch
import torch.nn.functional as F
def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 -mask) * torch.sqrt(distances)

    return distances

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k


    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()

    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


# Cell
def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl = F.relu(tl)
    triplet_loss = tl.mean()

    return triplet_loss

def batch_hard_semi_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    min_dist_mask = pairwise_dist > hardest_positive_dist

    mask_anchor_negative = mask_anchor_negative * min_dist_mask

    min_margin_dist_mask = pairwise_dist < (hardest_positive_dist + margin)

    mask_anchor_negative = mask_anchor_negative * min_margin_dist_mask

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl = F.relu(tl)
    triplet_loss = tl.mean()

    return triplet_loss



def batch_allpos_semi_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    anchor_negative_dist = mask_anchor_negative * pairwise_dist

    positive_dist_list = []
    negative_dist_list = []

    for i in range(len(anchor_positive_dist)):
        pos_index = torch.where(anchor_positive_dist[i] != 0, 1, 0)
        for p_i in pos_index.nonzero():
            positive_dist_list.append(anchor_positive_dist[i][p_i])

            neg_dist_mask = anchor_negative_dist[i] > anchor_positive_dist[i][p_i]
            anchor_negative_temp = anchor_negative_dist[i] * neg_dist_mask

            neg_dist_mask = anchor_negative_dist[i] < anchor_positive_dist[i][p_i] + margin
            anchor_negative_temp = anchor_negative_temp * neg_dist_mask

            if len(anchor_negative_temp.nonzero()) != 0:
                negative_dist_list.append(anchor_negative_temp[anchor_negative_temp.nonzero()[0]])
            else:
                negative_dist_list.append(anchor_negative_dist[i].max().unsqueeze(dim=0))

    hardest_positive_dist = torch.stack(positive_dist_list)
    hardest_negative_dist = torch.stack(negative_dist_list)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl = F.relu(tl)
    triplet_loss = tl.mean()

    return triplet_loss




def batch_hard_semi_adapted_triplet_loss(labels, embeddings, margin, lamda, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, positive_index = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    min_dist_mask = pairwise_dist > hardest_positive_dist

    mask_anchor_negative = mask_anchor_negative * min_dist_mask

    min_margin_dist_mask = pairwise_dist < (hardest_positive_dist + margin)

    mask_anchor_negative = mask_anchor_negative * min_margin_dist_mask

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, negative_index = anchor_negative_dist.min(1, keepdim=True)

    # measure l matbh

    # TODO: DT anchor average 구하기
    dt_list = []
    ds_list = []

    for li in range(8):
        label_index = torch.where(labels == li, 1, 0)
        neg_label_index = torch.where(labels != li, 1, 0)

        pos_result = torch.matmul(torch.unsqueeze(label_index, dim=0).float(), embeddings)

        embedding_pos_avg = torch.squeeze(pos_result / (label_index.nonzero().size()[0]+0.000000000000001))

        # print("CHECL label index : ", torch.unsqueeze(label_index, dim=0))
        # print("CHEKc embedding pos ave : ", embedding_pos_avg)

        neg_result = torch.matmul(torch.unsqueeze(neg_label_index, dim=0).float(), embeddings)
        embedding_neg_avg = torch.squeeze(neg_result / (neg_label_index.nonzero().size()[0]+0.000000000000001))

        dt_list.append(torch.stack([embedding_pos_avg, embedding_pos_avg, embedding_neg_avg]))

        # TODO: DS average 구하기.
        ds_pos_list = []
        ds_neg_list = []
        # print("CHECK non zero : ", label_index.nonzero())
        # print("CHECK non zero size () : ", label_index.nonzero().size())

        for dsp_i in label_index.nonzero():
            ds_pos_list.append(embeddings[positive_index[dsp_i[0]][0]])
            ds_neg_list.append(embeddings[negative_index[dsp_i[0]][0]])
        if len(ds_pos_list) == 0:
            ds_pos_list.append(torch.zeros(128))
            ds_neg_list.append(torch.zeros(128))

        ds_pos_list = torch.stack(ds_pos_list)
        ds_neg_list = torch.stack(ds_neg_list)
        ds_pos = torch.mean(ds_pos_list, dim=0).cuda()
        ds_neg = torch.mean(ds_neg_list, dim=0).cuda()

        ds_list.append(torch.stack([embedding_pos_avg, ds_pos, ds_neg]))

    dt_embedding = torch.stack(dt_list)
    ds_embedding = torch.stack(ds_list)

    dist_embedding = dt_embedding - ds_embedding

    norm = torch.norm(dist_embedding, p=2, dim=(1,2))
    l_match = norm.sum()


    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl = F.relu(tl)
    triplet_loss = tl.mean()

    triplet_loss = triplet_loss + lamda * l_match

    return triplet_loss

def batch_adapted_triplet_loss(labels, embeddings, margin, lamda, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()

    random_mask = torch.randint(10, (mask_anchor_positive.size()[0], mask_anchor_positive.size()[1])).float().cuda()
    mask_anchor_positive = mask_anchor_positive * random_mask

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, positive_index = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    random_mask = torch.rand((mask_anchor_positive.size()[0], mask_anchor_positive.size()[1])).float().cuda()
    mask_anchor_negative = mask_anchor_negative * random_mask

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, negative_index = anchor_negative_dist.min(1, keepdim=True)

    # TODO: DT anchor average 구하기
    dt_list = []
    ds_list = []

    for li in range(8):
        label_index = torch.where(labels == li, 1, 0)
        neg_label_index = torch.where(labels != li, 1, 0)

        pos_result = torch.matmul(torch.unsqueeze(label_index, dim=0).float(), embeddings)

        embedding_pos_avg = torch.squeeze(pos_result / (label_index.nonzero().size()[0]+0.000000000000001))

        # print("CHECL label index : ", torch.unsqueeze(label_index, dim=0))
        # print("CHEKc embedding pos ave : ", embedding_pos_avg)

        neg_result = torch.matmul(torch.unsqueeze(neg_label_index, dim=0).float(), embeddings)
        embedding_neg_avg = torch.squeeze(neg_result / (neg_label_index.nonzero().size()[0]+0.000000000000001))

        dt_list.append(torch.stack([embedding_pos_avg, embedding_pos_avg, embedding_neg_avg]))

        # TODO: DS average 구하기.
        ds_pos_list = []
        ds_neg_list = []
        # print("CHECK non zero : ", label_index.nonzero())
        # print("CHECK non zero size () : ", label_index.nonzero().size())

        for dsp_i in label_index.nonzero():
            ds_pos_list.append(embeddings[positive_index[dsp_i[0]][0]])
            ds_neg_list.append(embeddings[negative_index[dsp_i[0]][0]])
        if len(ds_pos_list) == 0:
            ds_pos_list.append(torch.zeros(128))
            ds_neg_list.append(torch.zeros(128))

        ds_pos_list = torch.stack(ds_pos_list)
        ds_neg_list = torch.stack(ds_neg_list)
        ds_pos = torch.mean(ds_pos_list, dim=0).cuda()
        ds_neg = torch.mean(ds_neg_list, dim=0).cuda()

        ds_list.append(torch.stack([embedding_pos_avg, ds_pos, ds_neg]))

    dt_embedding = torch.stack(dt_list)
    ds_embedding = torch.stack(ds_list)

    dist_embedding = dt_embedding - ds_embedding

    norm = torch.norm(dist_embedding, p=2, dim=(1,2))
    l_match = norm.sum()

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl = F.relu(tl)
    triplet_loss = tl.mean()

    # print("CHECK truplet loss : ", triplet_loss)
    # print("CHECK l match : ", l_match)

    triplet_loss = triplet_loss + lamda * l_match

    return triplet_loss


# Cell
def batch_first_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist


    # shape (batch_size, 1)
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + margin
    tl = F.relu(tl)
    triplet_loss = tl.mean()

    return triplet_loss

# Cell
def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin



    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = F.relu(triplet_loss)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)
    num_valid_triplets = mask.sum()

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets
