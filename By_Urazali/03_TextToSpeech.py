import os
os.add_dll_directory(r'C:\Program Files\VideoLAN\VLC')

import vlc
p = vlc.MediaPlayer(r'C:\Users\User\Desktop\Hakaton\audio\bir.ogg')
print(p)
print(p.play())
p.play()