import torch
import torch.nn as nn
import torch.utils.data as td
import numpy
from torchvision import datasets, transforms, utils
import torch.backends.cudnn as cudnn
import torch.optim as optim
import datetime
import time
import sys
from SmallImageDatasetFolder import GrowingSmallDatasetFolder
from GrowingGenerator import GrowingGenerator
from GrowingDiscriminator import GrowingDiscriminator
from torch.nn import functional as F
import Visualization

#general
batch_size = 32
use_cuda = True
num_epochs = 10000
load_checkpoint = False
file_location = './mnist_growing_4/'
is_testing = False
num_tests = 1000
do_rgb = False
noisy_inputs = True
use_wrong_gt_sometimes = False

#nets
sampling_size = 128 #GAN sampling space
starting_size = 4 #inital resolution of generated images (power of 2) 4-8-16-32-64-128-256-512-1024
up_scales = 3 #numer of upscale sessions (final resolution is <starting_size * 2^up_scales>
generator_use_layer_noise = False #inject noise into every layer during training
discriminator_use_layer_noise = False #inject noise into every layer during training
discriminator_layer_size = 128 #number of channels for each internal block
generator_layer_size = 256 #number of channels for each internal block
discriminator_use_tanh = False #use tanh instead of relus for internal blocks
generator_use_tanh = True
use_deconv = False #use ConvTranspose instead of Upsample+Conv2d
generator_learning_steps_every_cycle=1 #each cycle train the generator this number of steps
discriminator_learning_steps_every_cycle=2 #each cycle train the discriminator this number of steps
discriminator_checkpoint = "checkpoint_discriminator_net_at_epoc_11.pth"
generator_checkpoint = "checkpoint_generator_net_at_epoc_11.pth"
use_longer_network = True
discriminator_use_group_norm_instead_of_batchnorm = False
generator_use_group_norm_instead_of_batchnorm = True

#SGD
use_rmsprop_instead_of_adam = True
learning_rate = 0.0001
rmsprop_alpha = 0.99
eps = 1e-8
weight_decay = 0
momentum = 0
rmsprop_centered=True
adam_beta1=0.5
adam_beta2=0.999
adam_amsgrad=True
learning_rate_scheduler_step_size=10
learning_rate_scheduler_gamma=0.1

#dataset
datasets_location = 'D:/Development/Datasets/'
#datasets_location = 'D:/Development/Python/Datasets/'
dataset_dir = 'mnist_png/training'
#datset_dir = 'giger' 'Image_Celeb_Small' 'giger_beksinski'
images_multiplier = 1 #for small dataset this tells how many times each image is reused for each epoch
upright = True #force dataset augmentation to keep orientation of the image (disable random flips and force crops to be centered)
preload = False #useful for small dataset, preload all images

#visualization
vis_title = 'GrowingGAN'
generator_iter_plot = None
discriminator_iter_plot = None
epoch_plot = None
generator_sample_plot = None


#generate a sample with the generator given it's class id and sampling (yeah, it's not really noise)
def generate_class(max_classes, number, fixed_noise, generator, device):
    if max_classes > 1:
        x = numpy.full((fixed_noise.shape[0],max_classes), -1.0)
        for i in range(fixed_noise.shape[0]):
            x[i][number] = 1.0
        x1 = torch.from_numpy(x).float().to(device)
        input = torch.cat((x1, fixed_noise), 1)
    else:
        input = fixed_noise
    return generator(input)

#generate num_tests random samples for each class.
def test(generator, num_classes, device, num_tests):
    global sampling_size
    with torch.no_grad():
        for step in range(num_tests):
            if (num_classes > 1):
                noise = torch.randn(1, sampling_size, device=device)
                row = fake = generate_class(num_classes, 0, noise, generator, device)
                utils.save_image(fake, file_location + 'fake_sample%03d_step_%03d.png' % (0, step), normalize=True)
                for i in range(num_classes - 1):
                    fake = generate_class(num_classes, i + 1, noise, generator, device)
                    utils.save_image(fake, file_location + 'fake_sample%03d_step_%03d.png' % (i + 1, step), normalize=True)
                    row = torch.cat((row, fake), 3)
                utils.save_image(row, file_location + 'fake_sample_all_step_%03d.png' % (step), normalize=True)
            else:
                noise = torch.randn(1, sampling_size, device=device)
                fake = generate_class(1, 0, noise, generator, device)
                utils.save_image(fake, file_location + 'fake_sample_step_%03d.png' % (step), normalize=True)

#setup noisy ground truth for the GAN
def setup_groundtruth(batch_size, num_classes, goundtruth):
    global noisy_inputs
    # create expected discriminator outputs
    no_labels = numpy.full((batch_size, num_classes), -1)
    no_labels = torch.from_numpy(no_labels).float()
    gt = goundtruth.view(goundtruth.shape[0], 1)
    gt_one_hot = torch.FloatTensor(gt.size(0), num_classes).fill_(-1)
    gt_one_hot = gt_one_hot.scatter_(1, gt, 1)
    if num_classes > 1:
        if noisy_inputs:# add noise to expected class
            rd = (torch.randn((batch_size, num_classes * 2)) * 0.05) + 1.0
            gt_real = torch.cat((gt_one_hot, no_labels), 1) * rd
        else:
            gt_real = torch.cat((gt_one_hot, no_labels), 1)
    else:
        # add noise to expected class
        one = numpy.ones((batch_size, 1))
        if noisy_inputs:
            rd = (torch.randn((batch_size, 1)) * 0.05) + 1.0
            gt_real = torch.from_numpy(one).float() * rd
        else:
            gt_real = torch.from_numpy(one).float()
    if num_classes > 1:
        if noisy_inputs:# add noise to expected class
            rd = (torch.randn((batch_size, num_classes * 2)) * 0.05) + 1.0
            gt_fake = torch.cat((no_labels, gt_one_hot), 1) * rd
        else:
            gt_fake = torch.cat((no_labels, gt_one_hot), 1)
    else:
        # add noise to expected class
        mone = numpy.full((batch_size, 1), -1)
        if noisy_inputs:
            rd = (torch.randn((batch_size, 1)) * 0.05) + 1.0
            gt_fake = torch.from_numpy(mone).float() * rd
        else:
            gt_fake = torch.from_numpy(mone).float()
    return gt_one_hot, gt_real, gt_fake

def init_weights_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

#used to clip discriminator wheights (ala Wesserstein GAN)
def weights_clipping(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.clamp_(-0.01, 0.01)

#get a dataset element, if in transition phase generate one that gradually mix two resolutions
def get_dataset_element(dataset_iterator, is_transitioning, alpha):
    data = next(dataset_iterator)
    sample_big = data[0]
    if is_transitioning:
        sample_small = F.max_pool2d(sample_big, 2, 2)
        sample_small = F.interpolate(sample_small, size=None, scale_factor=2, mode='nearest')
        sample_big = sample_big * alpha + sample_small * (1.0 - alpha)
    return sample_big, data[1]

#train the network
def train(dataset,
          step,
          epoch,
          generator,
          discriminator,
          device,
          optimizer_g,
          optimizer_d,
          loss_criterion,
          fixed_noise, #fixed noise used for visualization every x steps
          generator_step,
          discriminator_step,
          is_transition_epoch #is a transitioning epoch
          ):
    global batch_size, file_location, sampling_size, generator_iter_plot, discriminator_iter_plot,\
        generator_sample_plot, vis_title, use_wrong_gt_sometimes

    last_time = time.time()

    running_discriminator_loss = 0.0
    running_discriminator_loss_n = 0
    running_generator_loss = 0.0
    running_generator_loss_n = 0
    epoch_generator_loss = 0.0
    epoch_generator_loss_n = 0
    epoch_discriminator_loss = 0.0
    epoch_discriminator_loss_n = 0
    last_generator_loss = 0.0
    last_discriminator_loss = 0.0

    dataset_loader = td.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    dataset_iterator = dataset_loader.__iter__()
    total_num_batches = len(dataset) // dataset_loader.batch_size
    num_classes = len(dataset.classes)

    discriminator.train()
    generator.train()

    num = 0
    while((num+discriminator_step+generator_step)<total_num_batches):

        step = step + 1
        alpha=1.0
        if is_transition_epoch: #change nets and dataset alpha (smoothing between resolution increases)
            alpha = num/total_num_batches
            generator.set_alpha(alpha)
            discriminator.set_alpha(alpha)


        discriminator.set_transition_time(is_transition_epoch) #enable discriminator learning in correct mode
        generator.set_transition_time(is_transition_epoch)  # enable generator learning in correct mode

        #for p in generator.parameters():
        #    p.requires_grad = False

        for i in range(discriminator_step): #train the discriminator batch
            discriminator.zero_grad()
            generator.zero_grad()

            image_cpu, gt = get_dataset_element(dataset_iterator, is_transition_epoch, alpha)  #fetch dataset batch
            num = num + 1
            image = image_cpu.to(device)

            #create expected discriminator outputs and generator input
            gt_one_hot, gt_real, gt_fake = setup_groundtruth(batch_size, num_classes, gt)
            gt_one_hot = gt_one_hot.to(device)

            #wrong assumption every tot batch, should help regularization
            if use_wrong_gt_sometimes:
                if ((i % discriminator_step) == (discriminator_step//2)):
                    discriminator_gt_real = gt_fake.to(device)
                    discriminator_gt_fake = gt_real.to(device)
                else:
                    discriminator_gt_real = gt_real.to(device)
                    discriminator_gt_fake = gt_fake.to(device)
            else:
                discriminator_gt_real = gt_real.to(device)
                discriminator_gt_fake = gt_fake.to(device)

            #train the disciminator with real data
            output = discriminator(image) #run the disciminator
            loss_real = loss_criterion(output, discriminator_gt_real)  #compute loss
            loss_real.backward()

            #generate generator random input
            noise = torch.randn(batch_size, sampling_size, device=device)
            if num_classes > 1: #combine it with condition if more than one class
                noise = torch.cat((gt_one_hot, noise), 1) #class
            #now train the discriminator with generated data
            fake = generator(noise)
            output = discriminator(fake.detach()) #run the disciminator on fake data.
            loss_fake = loss_criterion(output, discriminator_gt_fake) #compute loss
            loss_fake.backward()

            optimizer_d.step()  # do a step of the discriminator optimizer

            err_discriminator = (loss_real + loss_fake)
            running_discriminator_loss = running_discriminator_loss + err_discriminator.item()
            running_discriminator_loss_n = running_discriminator_loss_n + 1

            # Weight clipping
            discriminator.apply(weights_clipping)

        #for p in discriminator.parameters():
        #    p.requires_grad = False

        for i in range(generator_step): #and finally train the generator
            image_cpu, gt = get_dataset_element(dataset_iterator, is_transition_epoch, alpha)
            num = num + 1
            image = image_cpu.to(device)

            #create expected discriminator outputs and generator input
            gt_one_hot, gt_real, _ = setup_groundtruth(batch_size, num_classes, gt)
            gt_one_hot = gt_one_hot.to(device)
            generator_gt_real = gt_real.to(device)

            discriminator.zero_grad()
            generator.zero_grad()

            noise = torch.randn(batch_size, sampling_size, device=device)
            if num_classes > 1:  # combine it with condition if more than one class
                noise = torch.cat((gt_one_hot, noise), 1)  # class
            fake = generator(noise)
            output = discriminator(fake)
            err_generator = loss_criterion(output, generator_gt_real)
            err_generator.backward()
            optimizer_g.step()
            running_generator_loss = running_generator_loss + err_generator.item()
            running_generator_loss_n = running_generator_loss_n + 1;

        if running_generator_loss_n > 0:
            running_generator_loss = running_generator_loss / running_generator_loss_n
            last_generator_loss = (last_generator_loss + running_generator_loss) / 2.0
            epoch_generator_loss = epoch_generator_loss + running_generator_loss
            epoch_generator_loss_n = epoch_generator_loss_n + 1
            if generator_iter_plot is not None:
                Visualization.update_vis_plot1(step, running_generator_loss, generator_iter_plot, 'append')
        if running_discriminator_loss_n > 0:
            running_discriminator_loss = running_discriminator_loss / running_discriminator_loss_n
            last_discriminator_loss = (last_discriminator_loss + running_discriminator_loss) / 2.0
            epoch_discriminator_loss = epoch_discriminator_loss + running_discriminator_loss
            epoch_discriminator_loss_n = epoch_discriminator_loss_n + 1
            if discriminator_iter_plot is not None:
                Visualization.update_vis_plot1(step, running_discriminator_loss, discriminator_iter_plot, 'append')
        time_spent = time.time() - last_time
        batch_time = time_spent / num
        time_to_finish = batch_time * (total_num_batches - num)
        print("epoch {} of {} at {}%, to finish epoch: {}".format(epoch, num_epochs, (num / total_num_batches) * 100, str(datetime.timedelta(seconds=time_to_finish))))
        print("generator loss {}, discriminator loss {}".format(running_generator_loss, running_discriminator_loss))
        running_discriminator_loss = 0.0
        running_generator_loss = 0.0
        running_generator_loss_n = 0
        running_discriminator_loss_n = 0

        if (step % 10 == 9):
            generator.eval()
            with torch.no_grad():
                row = fake = generate_class(num_classes, 0, fixed_noise, generator, device)
                for i in range(num_classes-1):
                    fake = generate_class(num_classes, i+1, fixed_noise, generator, device)
                    row = torch.cat((row, fake), 0)
                Visualization.plot_images(row.cpu(), 8, generator_sample_plot, vis_title, 'fake_sample_all_step%03d.png'% (step))
            torch.save(generator.state_dict(), file_location + 'checkpoint_generator_net.pth')
            torch.save(discriminator.state_dict(), file_location + 'checkpoint_discriminator_net.pth')
            generator.train()

    epoch_generator_loss = epoch_generator_loss / epoch_generator_loss_n
    epoch_discriminator_loss = epoch_discriminator_loss / epoch_discriminator_loss_n
    dataset_iterator.__del__()
    return step, epoch_generator_loss, epoch_discriminator_loss

def run():
    global batch_size, use_cuda, num_epochs, starting_size, load_checkpoint, discriminator_checkpoint, generator_checkpoint,\
        file_location,is_testing, num_tests, do_rgb, generator_use_layer_noise, discriminator_use_layer_noise, \
        discriminator_layer_size, generator_layer_size, generator_use_tanh, use_deconv, sampling_size, vis_title, epoch_plot,\
        generator_iter_plot,discriminator_iter_plot,generator_sample_plot,datasets_location, up_scales,images_multiplier,\
        upright, preload, dataset_dir, use_rmsprop_instead_of_adam, learning_rate, rmsprop_alpha, eps, weight_decay, \
        momentum, rmsprop_centered, adam_beta1, adam_beta2, adam_amsgrad, learning_rate_scheduler_step_size, \
        learning_rate_scheduler_gamma, generator_learning_steps_every_cycle, discriminator_learning_steps_every_cycle,\
        use_longer_network, discriminator_use_group_norm_instead_of_batchnorm, generator_use_group_norm_instead_of_batchnorm, \
        discriminator_use_tanh

    dataset = GrowingSmallDatasetFolder(datasets_location + dataset_dir, init_size=starting_size, growt_number=up_scales, images_multiplier=images_multiplier, upright=upright, do_rgb=do_rgb, preload=preload)

    vis_title = 'GrowingGAN on ' + dataset_dir
    generator_iter_plot = Visualization.create_vis_plot1('Iteration', 'Loss', vis_title, ['Generator Loss'])
    discriminator_iter_plot = Visualization.create_vis_plot1('Iteration', 'Loss', vis_title, ['Discriminator Loss'])
    epoch_plot = Visualization.create_vis_plot2('Epoch', 'Loss', vis_title, ['Generator Loss', 'Discriminator Loss'])
    generator_sample_plot = Visualization.create_plot_images(vis_title)

    num_classes = len(dataset.classes)

    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            use_cuda = False
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print("device={}".format(device))
    #if use_cuda:
    #    cudnn.benchmark = True

    fixed_noise = torch.randn(8, sampling_size, device=device)

    if do_rgb is True:
        channels = 3
    else:
        channels = 1

    discriminator_net = GrowingDiscriminator(device, input_channels=channels,output_size=num_classes,
                                             starting_size=starting_size, use_layer_noise=discriminator_use_layer_noise,
                                             layer_channel=discriminator_layer_size, longer_network=use_longer_network,
                                             use_groupnorm=discriminator_use_group_norm_instead_of_batchnorm, use_tanh=discriminator_use_tanh).to(device)
    generator_net = GrowingGenerator(device, input_class_size=num_classes, output_channels=channels, starting_size=starting_size,
                                     use_tanh_instead_of_relu=generator_use_tanh, use_layer_noise=generator_use_layer_noise,
                                     use_deconv_instead_of_upsample=use_deconv,layer_channel=generator_layer_size,
                                     sampling_size=sampling_size, longer=use_longer_network, use_groupnorm=generator_use_group_norm_instead_of_batchnorm).to(device)

    discriminator_net.apply(init_weights_normal)
    generator_net.apply(init_weights_normal)

    #load latest checkpoint
    if(load_checkpoint is True) and use_cuda is True:
        discriminator_net.load_state_dict(torch.load(file_location + discriminator_checkpoint))
        generator_net.load_state_dict(torch.load(file_location + generator_checkpoint))
    if (load_checkpoint is True) and use_cuda is False:
        discriminator_net.load_state_dict(torch.load(file_location + discriminator_checkpoint, map_location='cpu'))
        generator_net.load_state_dict(torch.load(file_location + generator_checkpoint, map_location='cpu'))

    criterion = nn.MSELoss()

    if use_rmsprop_instead_of_adam:
        discriminator_optimizer = optim.RMSprop(discriminator_net.parameters(), lr=learning_rate, alpha=rmsprop_alpha,
                                                eps=eps, weight_decay=weight_decay, momentum=momentum,
                                                centered=rmsprop_centered)
        generator_optimizer = optim.RMSprop(generator_net.parameters(), lr=learning_rate, alpha=rmsprop_alpha,
                                            eps=eps, weight_decay=weight_decay, momentum=momentum,
                                            centered=rmsprop_centered)
    else:
        discriminator_optimizer = optim.Adam(discriminator_net.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2),
                                             weight_decay=weight_decay, amsgrad=adam_amsgrad, eps=eps)
        generator_optimizer = optim.Adam(generator_net.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2),
                                         weight_decay=weight_decay, amsgrad=adam_amsgrad, eps=eps)

    step = 0
    #decrease learning rate every 10 epochs
    scheduler_d = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=learning_rate_scheduler_step_size, gamma=learning_rate_scheduler_gamma)
    scheduler_g = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=learning_rate_scheduler_step_size, gamma=learning_rate_scheduler_gamma)

    if is_testing is False:
        for epoch in range(num_epochs):
            if (epoch % 2 == 0): #every even epoch is a stabilizing epoch
                is_transitioning = False
                dataset.set_transitioning(False)
            else: #every odd epoch increase size
                ret = dataset.increase_size() #untill you reach a max set size
                is_transitioning = False
                if(ret==True):
                    is_transitioning = True
                    dataset.set_transitioning(True)
                    generator_net.increase_size()
                    discriminator_net.increase_size()

            step, epoch_generator_loss, epoch_discriminator_loss = train(dataset=dataset,
                                                                          step=step,
                                                                          epoch=epoch,
                                                                          generator=generator_net,
                                                                          discriminator=discriminator_net,
                                                                          device=device,
                                                                          optimizer_g=generator_optimizer,
                                                                          optimizer_d=discriminator_optimizer,
                                                                          loss_criterion=criterion,
                                                                          fixed_noise=fixed_noise,
                                                                          generator_step=generator_learning_steps_every_cycle,
                                                                          discriminator_step=discriminator_learning_steps_every_cycle,
                                                                          is_transition_epoch=is_transitioning)

            Visualization.update_vis_plot2(epoch, epoch_generator_loss, epoch_discriminator_loss, epoch_plot, 'append')

            scheduler_d.step()
            scheduler_g.step()
            print("EPOCH generator loss {}, discriminator loss {}".format(epoch_generator_loss,epoch_discriminator_loss))
            torch.save(generator_net.state_dict(), file_location + 'checkpoint_generator_net_at_epoc_{}.pth'.format(epoch))
            torch.save(discriminator_net.state_dict(), file_location + 'checkpoint_discriminator_net_at_epoc_{}.pth'.format(epoch))
    else:
        test(generator_net, num_classes, device, num_tests)


if __name__ == '__main__':
    run()