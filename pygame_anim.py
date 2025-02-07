import pygame
from train import *
import math
import random

pygame.init()
screen = pygame.display.set_mode((800, 600))
running = True
dragging = None

dots = []
springs = []

def display():
    screen.fill((255, 255, 255))
    for spring in springs:
        dot1, dot2, _ = spring
        pygame.draw.line(screen, (100, 100, 100), dots[dot1][:2], dots[dot2][:2], 1)
    for i, dot in enumerate(dots):
        color = (200, 20, 20)
        if i >= len(data_numpy):
            color = (20, 20, 200)
        pygame.draw.circle(screen, color, dot[:2], 5)

    pygame.display.update()

def events():
    global running
    global dragging
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            for i, (px, py, _, _) in enumerate(dots):
                if (x-px)**2 + (y-py)**2 < 25:
                    dragging = i
                    break
        if event.type == pygame.MOUSEBUTTONUP:
            dragging = None
    if dragging is not None:
        x, y = pygame.mouse.get_pos()
        dots[dragging] = (x, y, 0, 0)

def compute():
    for dot1, dot2, target in springs:
        x1, y1, vx1, vy1 = dots[dot1]
        x2, y2, vx2, vy2 = dots[dot2]
        dx = x2 - x1
        dy = y2 - y1
        d = math.sqrt(dx**2 + dy**2)
        if d < 1:
            d = 1
        diff = d - target
        diffx = diff * dx / d
        diffy = diff * dy / d
        vx1 += diffx * 0.005
        vy1 += diffy * 0.005
        vx2 -= diffx * 0.005
        vy2 -= diffy * 0.005
        dots[dot1] = (x1, y1, vx1, vy1)
        dots[dot2] = (x2, y2, vx2, vy2)
    
    for i, dot in enumerate(dots):
        # gravity towards the center
        dots[i] = (dot[0], dot[1], dot[2]+(400-dot[0])*0.001, dot[3]+(300-dot[1])*0.001)
        dot = dots[i]

        # repel each other
        for j, dot2 in enumerate(dots):
            if i == j:
                continue
            dx = dot2[0] - dot[0]
            dy = dot2[1] - dot[1]
            d = math.sqrt(dx**2 + dy**2)
            if d < 1:
                d = 1
            dots[i] = (dot[0], dot[1], dot[2] - dx/d*0.1, dot[3] - dy/d*0.1)
        dot = dots[i]

        # friction
        dots[i] = (dot[0], dot[1], dot[2]*0.9, dot[3]*0.9)
        # update position
        x, y, vx, vy = dots[i]
        x += vx
        y += vy
        if y > 600:
            y = 600
            vy = -vy
        if x > 800:
            x = 800
            vx = -vx
        if x < 0:
            x = 0
            vx = -vx
        if y < 0:
            y = 0
            vy = -vy
        dots[i] = (x, y, vx, vy)



if __name__ == '__main__':
    data = load_data()
    data_numpy = data.to_numpy()
    data_numpy = to_relative(data_numpy)

    for i in range(0, data_numpy.shape[0]):
        dots.append((random.randint(0, 800), random.randint(0, 600), 0, 0))
    for i in range(0, data_numpy.shape[1]):
        dots.append((random.randint(0, 800), random.randint(0, 600), 0, 0))
    
    for i in range(0, data_numpy.shape[0]):
        for j in range(0, data_numpy.shape[1]):
            if not np.isnan(data_numpy[i][j]):
                springs.append((i, j+data_numpy.shape[0], data_numpy[i][j]*200))

    clock = pygame.time.Clock()
    while running:
        events()
        compute()
        display()
        clock.tick(60)

    pygame.quit()