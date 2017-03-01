# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:05:00 2013

@author: Leo
"""
from __future__ import print_function
import pygame
from sys import exit
from pygame.locals import *
from gameRole import *
import random

import tensorflow as tf
import cv2
import numpy as np
from collections import deque


GAME = 'plane' # the name of the game being played for log files
ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.5 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

# 初始化游戏
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('飞机大战')

# 载入游戏音乐
bullet_sound = pygame.mixer.Sound('resources/sound/bullet.wav')
enemy1_down_sound = pygame.mixer.Sound('resources/sound/enemy1_down.wav')
game_over_sound = pygame.mixer.Sound('resources/sound/game_over.wav')
bullet_sound.set_volume(0.3)
enemy1_down_sound.set_volume(0.3)
game_over_sound.set_volume(0.3)
pygame.mixer.music.load('resources/sound/game_music.wav')
pygame.mixer.music.play(-1, 0.0)
pygame.mixer.music.set_volume(0.25)

# 载入背景图
background = pygame.image.load('resources/image/background.png').convert()
game_over = pygame.image.load('resources/image/gameover.png')

filename = 'resources/image/shoot.png'
plane_img = pygame.image.load(filename)

# 设置玩家相关参数
player_rect = []
player_rect.append(pygame.Rect(0, 99, 102, 126))        # 玩家精灵图片区域
player_rect.append(pygame.Rect(165, 360, 102, 126))
player_rect.append(pygame.Rect(165, 234, 102, 126))     # 玩家爆炸精灵图片区域
player_rect.append(pygame.Rect(330, 624, 102, 126))
player_rect.append(pygame.Rect(330, 498, 102, 126))
player_rect.append(pygame.Rect(432, 624, 102, 126))
player_pos = [200, 600]
player = Player(plane_img, player_rect, player_pos)

# 定义子弹对象使用的surface相关参数
bullet_rect = pygame.Rect(1004, 987, 9, 21)
bullet_img = plane_img.subsurface(bullet_rect)

# 定义敌机对象使用的surface相关参数
enemy1_rect = pygame.Rect(534, 612, 57, 43)
enemy1_img = plane_img.subsurface(enemy1_rect)
enemy1_down_imgs = []
enemy1_down_imgs.append(plane_img.subsurface(pygame.Rect(267, 347, 57, 43)))
enemy1_down_imgs.append(plane_img.subsurface(pygame.Rect(873, 697, 57, 43)))
enemy1_down_imgs.append(plane_img.subsurface(pygame.Rect(267, 296, 57, 43)))
enemy1_down_imgs.append(plane_img.subsurface(pygame.Rect(930, 697, 57, 43)))

enemies1 = pygame.sprite.Group()

# 存储被击毁的飞机，用来渲染击毁精灵动画
enemies_down = pygame.sprite.Group()

shoot_frequency = 0
enemy_frequency = 0

player_down_index = 16

score = 0

clock = pygame.time.Clock()

running = True

sess = tf.InteractiveSession()
s, readout, h_fc1 = createNetwork()
# trainNetwork(s, readout, h_fc1, sess)
a = tf.placeholder("float", [None, ACTIONS])
y = tf.placeholder("float", [None])
readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
cost = tf.reduce_mean(tf.square(y - readout_action))
train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

# store the previous observations in replay memory
D = deque()

x_t = pygame.surfarray.array3d(pygame.display.get_surface())
x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

# saving and loading networks
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state("saved_networks")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
    print("Could not find old network weights")

# start training
epsilon = INITIAL_EPSILON
t = 0
while running:
    # 控制游戏最大帧率为60
    clock.tick(120)
    reward = 0.1
    # 控制发射子弹频率,并发射子弹
    if not player.is_hit:
        if shoot_frequency % 15 == 0:
            bullet_sound.play()
            player.shoot(bullet_img)
        shoot_frequency += 1
        if shoot_frequency >= 15:
            shoot_frequency = 0

    # 生成敌机
    if enemy_frequency % 10 == 0:
        enemy1_pos = [random.randint(0, SCREEN_WIDTH - enemy1_rect.width), 0]
        enemy1 = Enemy(enemy1_img, enemy1_down_imgs, enemy1_pos)
        enemies1.add(enemy1)
    enemy_frequency += 1
    if enemy_frequency >= 100:
        enemy_frequency = 0

    # 移动子弹，若超出窗口范围则删除
    for bullet in player.bullets:
        bullet.move()
        if bullet.rect.bottom < 0:
            player.bullets.remove(bullet)

    # 移动敌机，若超出窗口范围则删除
    for enemy in enemies1:
        enemy.move()
        # 判断玩家是否被击中
        if pygame.sprite.collide_circle(enemy, player):
            enemies_down.add(enemy)
            enemies1.remove(enemy)
            player.is_hit = True
            game_over_sound.play()
            break
        if enemy.rect.top > SCREEN_HEIGHT:
            enemies1.remove(enemy)

    # 将被击中的敌机对象添加到击毁敌机Group中，用来渲染击毁动画
    enemies1_down = pygame.sprite.groupcollide(enemies1, player.bullets, 1, 1)
    for enemy_down in enemies1_down:
        enemies_down.add(enemy_down)

    # 绘制背景
    screen.fill(0)
    screen.blit(background, (0, 0))

    # 绘制玩家飞机
    if not player.is_hit:
        screen.blit(player.image[player.img_index], player.rect)
        # 更换图片索引使飞机有动画效果
        player.img_index = shoot_frequency // 8
    else:
        player.img_index = player_down_index // 8
        screen.blit(player.image[player.img_index], player.rect)
        player_down_index += 1
        reward = -1
        running = False
        # if player_down_index > 47:
        #     running = False

    # 绘制击毁动画
    for enemy_down in enemies_down:
        if enemy_down.down_index == 0:
            enemy1_down_sound.play()
        if enemy_down.down_index > 7:
            enemies_down.remove(enemy_down)
            score += 1000
            continue
        screen.blit(enemy_down.down_imgs[enemy_down.down_index // 2], enemy_down.rect)
        enemy_down.down_index += 1
        reward += 1

    # 绘制子弹和敌机
    player.bullets.draw(screen)
    enemies1.draw(screen)

    # 绘制得分
    score_font = pygame.font.Font(None, 36)
    score_text = score_font.render(str(score), True, (128, 128, 128))
    text_rect = score_text.get_rect()
    text_rect.topleft = [10, 10]
    screen.blit(score_text, text_rect)

    # 更新屏幕
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
            
    # 监听键盘事件
    key_pressed = pygame.key.get_pressed()
    # 若玩家被击中，则无效
    # if not player.is_hit:
    #     if key_pressed[K_w] or key_pressed[K_UP]:
    #         player.moveUp()
    #     if key_pressed[K_s] or key_pressed[K_DOWN]:
    #         player.moveDown()
    #     if key_pressed[K_a] or key_pressed[K_LEFT]:
    #         player.moveLeft()
    #     if key_pressed[K_d] or key_pressed[K_RIGHT]:
    #         player.moveRight()
    # choose an action epsilon greedily

    readout_t = readout.eval(feed_dict={s: [s_t]})[0]
    a_t = np.zeros([ACTIONS])
    action_index = 0
    if t % FRAME_PER_ACTION == 0:
        if random.random() <= epsilon:
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[random.randrange(ACTIONS)] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1
    else:
        a_t[0] = 1  # do nothing

    # scale down epsilon
    if epsilon > FINAL_EPSILON and t > OBSERVE:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 若玩家被击中，则无效
    if not player.is_hit:
        if a_t[1] == 1:
            player.moveLeft()
        if a_t[2] == 1:
            player.moveRight()
        # if not player.is_hit:
        #     if key_pressed[K_w] or key_pressed[K_UP]:
        #         player.moveUp()
        #     if key_pressed[K_s] or key_pressed[K_DOWN]:
        #         player.moveDown()
        #     if key_pressed[K_a] or key_pressed[K_LEFT]:
        #         player.moveLeft()
        #     if key_pressed[K_d] or key_pressed[K_RIGHT]:
        #         player.moveRight()


    # run the selected action and observe next state and reward
    x_t1_colored = pygame.surfarray.array3d(pygame.display.get_surface())
    r_t = reward
    terminal = running
    x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (80, 80, 1))
    # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
    s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

    # store the transition in D
    D.append((s_t, a_t, r_t, s_t1, terminal))
    if len(D) > REPLAY_MEMORY:
        D.popleft()

    # only train if done observing
    if t > OBSERVE:
        # sample a minibatch to train on 抽取小批量样本进行训练
        minibatch = random.sample(D, BATCH)

        # get the batch variables
        s_j_batch = [d[0] for d in minibatch]  #当前状态
        a_batch = [d[1] for d in minibatch]     #动作
        r_batch = [d[2] for d in minibatch]     #奖励
        s_j1_batch = [d[3] for d in minibatch]  #返回下一个状态

        y_batch = []
        readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
        for i in range(0, len(minibatch)):
            terminal = minibatch[i][4]
            # if terminal, only equals reward
            if terminal:
                y_batch.append(r_batch[i])
            else:
                y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

        # perform gradient step
        train_step.run(feed_dict={
            y: y_batch,
            a: a_batch,
            s: s_j_batch}
        )

    # update the old values
    s_t = s_t1
    t += 1

    # save progress every 10000 iterations
    if t % 10000 == 0:
        saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)

    # print info
    state = ""
    if t <= OBSERVE:
        state = "observe"
    elif t > OBSERVE and t <= OBSERVE + EXPLORE:
        state = "explore"
    else:
        state = "train"

    print("TIMESTEP", t, "/ STATE", state, \
          "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
          "/ Q_MAX %e" % np.max(readout_t))
    if running == False:
        # 设置玩家相关参数
        player_rect = []
        player_rect.append(pygame.Rect(0, 99, 102, 126))  # 玩家精灵图片区域
        player_rect.append(pygame.Rect(165, 360, 102, 126))
        player_rect.append(pygame.Rect(165, 234, 102, 126))  # 玩家爆炸精灵图片区域
        player_rect.append(pygame.Rect(330, 624, 102, 126))
        player_rect.append(pygame.Rect(330, 498, 102, 126))
        player_rect.append(pygame.Rect(432, 624, 102, 126))
        player_pos = [200, 600]
        player = Player(plane_img, player_rect, player_pos)
        # 定义子弹对象使用的surface相关参数
        bullet_rect = pygame.Rect(1004, 987, 9, 21)
        bullet_img = plane_img.subsurface(bullet_rect)
        # 定义敌机对象使用的surface相关参数
        enemy1_rect = pygame.Rect(534, 612, 57, 43)
        enemy1_img = plane_img.subsurface(enemy1_rect)
        enemy1_down_imgs = []
        enemy1_down_imgs.append(plane_img.subsurface(pygame.Rect(267, 347, 57, 43)))
        enemy1_down_imgs.append(plane_img.subsurface(pygame.Rect(873, 697, 57, 43)))
        enemy1_down_imgs.append(plane_img.subsurface(pygame.Rect(267, 296, 57, 43)))
        enemy1_down_imgs.append(plane_img.subsurface(pygame.Rect(930, 697, 57, 43)))
        enemies1 = pygame.sprite.Group()
        # 存储被击毁的飞机，用来渲染击毁精灵动画
        enemies_down = pygame.sprite.Group()
        shoot_frequency = 0
        enemy_frequency = 0
        player_down_index = 16
        score = 0
        clock = pygame.time.Clock()
        running = True


font = pygame.font.Font(None, 48)
text = font.render('Score: '+ str(score), True, (255, 0, 0))
text_rect = text.get_rect()
text_rect.centerx = screen.get_rect().centerx
text_rect.centery = screen.get_rect().centery + 24
screen.blit(game_over, (0, 0))
screen.blit(text, text_rect)

while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    pygame.display.update()
