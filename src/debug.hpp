#ifdef DEBUG
#include <set>

void showImage(std::string const& windowName, cv::Mat const& image)
{
   static std::set<std::string> setOfWindows;
   if (!setOfWindows.count(windowName))
   {
      setOfWindows.insert(windowName);
   }
   cv::imshow(windowName, image);
   cv::waitKey(1);
}
#else
#define showImage(windowName, image)
#endif