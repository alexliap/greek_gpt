import mlx.core as mx
import mlx.nn as nn


def get_batch_sent(data, batch_size, block_size, tokenizer):
    batch_x = []
    batch_y = []
    ix = mx.random.randint(
        0,
        len(data),
        shape=[
            batch_size,
        ],
    )
    sec_ix = mx.random.uniform(
        0,
        1,
        shape=[
            batch_size,
        ],
    )
    for index, sec_index in zip(ix, sec_ix):
        size = len(data[index.item()])
        x = data[index.item()][int(sec_index.item() * size) :]
        x = mx.array(x)
        if len(x) <= block_size:
            in_x = mx.pad(x, (block_size - len(x) + 1) // 2, tokenizer.pad_token_id)[
                0:block_size
            ]
            y = mx.pad(x, (block_size + 1 - len(x) + 1) // 2, tokenizer.pad_token_id)[
                1 : block_size + 1
            ]
        else:
            in_x = x[0:block_size]
            y = x[1 : block_size + 1]
        batch_x.append(in_x)
        batch_y.append(y)

    batch_x = mx.stack(batch_x)
    batch_y = mx.stack(batch_y)

    yield batch_x, batch_y


def get_batch(data, batch_size, block_size, tokenizer):
    ix = mx.random.randint(
        0,
        len(data) - block_size,
        shape=[
            batch_size,
        ],
    )
    batch_x = mx.stack(mx.array([data[i.item() : i.item() + block_size] for i in ix]))
    batch_y = mx.stack(
        mx.array([data[i.item() + 1 : i.item() + block_size + 1] for i in ix])
    )

    return batch_x, batch_y


def loss_fn(model, x, y):
    out = model(x)
    return nn.losses.cross_entropy(out, y, reduction="mean")


def training_step(model, loss_and_grad_fn, x, y):
    # set model to training mode
    model.train(True)
    loss, grad = loss_and_grad_fn(model, x, y.reshape(-1))
    model.optim.update(model, grad)

    return loss


def validation_step(model, x, y):
    # set model to evaluation mode
    model.train(False)
    val_loss = loss_fn(model, x, y.reshape(-1))

    return val_loss
