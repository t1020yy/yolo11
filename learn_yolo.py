import glob

from ultralytics import YOLO
from monkey_patch import apply_proto_upsample_patch


PATH_TO_DATASET = './dust-particle-trajectories'

YOLO_MODEL_TOTRAIN = './yolo11n-seg.pt'

# PATH_TO_DATASET = r'D:\yy\phd\datasets\dust-particle-trajectories'
# YOLO_MODEL_TOTRAIN = r'D:\yy\phd\datasets\yolo11n-seg.pt'
EPOCHS_TO_TRAIN = 250
BATCH_SIZE = -1
IMG_SIZE_TOTRAIN = 640

RESUME_TRAINNING = False

if __name__ == '__main__':

  # Старая версия с 2000 изображений
  # train3 - тренировка на colab размер nano, 300 эпох
  # train4 - размер nano, 300 эпох
  # train5 - размер nano, 2000 эпох
  # train6 - размер small, 500 эпох, batch 12?
  # train7 - размер medium, 500 эпох, batch 7
  # train8 - размер nano, 500 эпох, bacth 20
  # train9 - размер nano, 500 эпох, bacth 10

  # Новая версия с 10_000 изображений
  # train - размер nano, 250 эпох, batch 16, img_size 640, in 8.189 hours
  # train2 - размер nano, 250 эпох, batch 32, img_size 640, 7.321 hours
  # train3 - размер small, 250 эпох, batch 20, img_size 640, 11.2 hours
  # train4 - размер medium, 250 эпох, batch 16, img_size 640, 20-22 hours
  # train5 - размер large, 250 эпох, batch 20, img_size 640, 2 hours
  # train6 - размер medium, 250 эпох, batch 7, img_size 960, 47 hours
  # train7 - размер small, 250 эпох, batch 12, img_size 960, ?? hours
  # train8 - размер small, 250 эпох, batch 6, img_size 1440, 51 hours
  # train9 - размер medium, 400 эпох, batch 7, img_size 960, 75.6 hours
  # train10 - размер small, 700 эпох, batch 20, img_size 640, ?? hours

  trains = glob.glob(f'{PATH_TO_DATASET}/runs1/train*')
  trains.sort()

  resume_training = RESUME_TRAINNING

  if len(trains) > 0 and RESUME_TRAINNING:
    train_numbers = [train.split('train')[-1] for train in trains]
    last_train_path = sorted([(int(num), train) for num, train in zip(train_numbers, trains) if num != ''])[-1][1]
    print(f'Выбираем последнюю тренировку: {last_train_path}')

    try:
      model = YOLO(f'{last_train_path}/weights/last.pt')

      # Определяем нужно ли продолжить тренировку
      epochs_to_train = int(model.ckpt['train_args']['epochs'])
      last_epoch_in_training = int(model.ckpt['train_results']['epoch'][-1])

      if not last_epoch_in_training < epochs_to_train:
        resume_training = False
        print(f'Для {last_train_path} обучение завершено')

    except FileNotFoundError as ex:
      print(f'При загрузки {last_train_path} произошла ошибка: {ex}')
      resume_training = False
  else:
    resume_training = False
    print(f'Предыдущих тренировок не найдено')


  if resume_training:
    print(f'Продолжаем тренировку c эпохи {last_epoch_in_training} для {epochs_to_train} эпох\n')
  else:
    print(f'Начинаем новую тренировку {YOLO_MODEL_TOTRAIN} для {EPOCHS_TO_TRAIN} эпох\n')
    model = YOLO(YOLO_MODEL_TOTRAIN)

  # model = YOLO('yolov8.yaml').to(device) # Train the model 
  # model.train(data=f'{PATH_TO_DATASET}parabola_local.yaml', epochs=50, device=device)

  apply_proto_upsample_patch(4)

  # Начинаем тренировку
  model.train(
      # data=f'./datasets/parabola.yaml',
      data = 'D:/yy/phd/datasets/parabola.yaml', 
      project=f'{PATH_TO_DATASET}/runs1',
      epochs=EPOCHS_TO_TRAIN,
      imgsz=IMG_SIZE_TOTRAIN,
      batch=BATCH_SIZE,
      cache='disk',
      single_cls=True,
      patience=0,
      resume=resume_training,
      mask_ratio = 1,
      retina_masks = True)
